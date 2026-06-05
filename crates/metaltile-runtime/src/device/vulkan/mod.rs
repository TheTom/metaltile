//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Vulkan compute runtime (`VULKAN_BACKEND_SPEC.md` — Phase 1).
//!
//! Mirrors `CudaDevice` / `HipDevice`: own a `VkDevice` + compute queue,
//! compile GLSL → SPIR-V via shaderc, build a compute pipeline, dispatch.
//! Phase 1 covers the elementwise smoke path (vector_add).
//!
//! ## Memory model
//!
//! For Phase-1 simplicity we use a single host-visible+device-local memory
//! type (the "BAR" path that integrated and modern desktop GPUs expose),
//! and `vkMapMemory` for host↔device transfer. The dedicated staging-buffer
//! + transfer-queue path (the "proper" decoupled DMA layout) is Phase 2 —
//! the only consequence on Phase 1 perf is that the read-back includes a
//! full PCIe round trip. For RX 9070 XT's resizable-BAR config, even the
//! Phase-1 path is direct VRAM access.
//!
//! ## Layout
//!
//! - SSBOs at `binding = 0..N` in `kernel.params` order (matches the
//!   emitter's `binding_plan`).
//! - One push-constant block carrying constexprs + the synthetic `_n_elems`
//!   (Elementwise) — total ≤128 bytes, well under the Vulkan guaranteed
//!   minimum.
//! - One descriptor set (`set = 0`), one compute pipeline per kernel.

mod ffi;

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

use metaltile_codegen::{CodegenBackend, GlslGenerator, spirv::GlslBindingPlan};
use metaltile_core::{dtype::DType, ir::Kernel};

use crate::error::MetalTileError;

use ffi::*;

const ENTRY_POINT: &[u8] = b"main\0";

/// Synthesize a Strided param's `_shape` or `_strides` companion buffer
/// (row-major) from the param's static shape. Used when the harness
/// doesn't supply explicit companion bytes.
fn synth_strided_meta(shape: &metaltile_core::shape::Shape, strides: bool) -> Vec<u8> {
    use metaltile_core::shape::Dim;
    let dims: Vec<u32> = (0..shape.rank())
        .map(|i| match shape.dim(i) {
            Some(Dim::Known(n)) => *n as u32,
            _ => 1,
        })
        .collect();
    let vals: Vec<u32> = if strides {
        let mut s = vec![1u32; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * dims[i + 1];
        }
        s
    } else {
        dims
    };
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn vk_check(res: VkResult, what: &str) -> Result<(), MetalTileError> {
    if res == VK_SUCCESS {
        return Ok(());
    }
    Err(MetalTileError::Dispatch(format!("{what}: VkResult={res}")))
}

/// A host-visible+coherent buffer backed by a single device allocation.
/// Drops the buffer + frees the memory when the host releases it.
pub struct VulkanBuffer<'d> {
    pub(crate) buffer: VkBuffer_,
    pub(crate) memory: VkDeviceMemory,
    pub(crate) size: u64,
    dev: &'d VulkanDevice,
}

impl VulkanBuffer<'_> {
    pub fn handle(&self) -> VkBuffer_ { self.buffer }
    pub fn size(&self) -> u64 { self.size }
}

impl Drop for VulkanBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            if self.buffer != VK_NULL_HANDLE {
                vkDestroyBuffer(self.dev.device, self.buffer, ptr::null());
            }
            if self.memory != VK_NULL_HANDLE {
                vkFreeMemory(self.dev.device, self.memory, ptr::null());
            }
        }
    }
}

/// A compiled compute pipeline + its descriptor-set layout + pipeline
/// layout. Owns the SPIR-V shader module.
pub struct VulkanPipeline {
    pub pipeline: VkPipeline_,
    pub layout: VkPipelineLayout,
    pub set_layout: VkDescriptorSetLayout,
    pub shader_module: VkShaderModule,
    pub push_constant_bytes: u32,
}

/// Top-level Vulkan compute device. Holds the instance, physical device,
/// logical device, compute queue, descriptor pool, and command pool.
pub struct VulkanDevice {
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    device: VkDevice,
    queue: VkQueue,
    queue_family_index: u32,
    descriptor_pool: VkDescriptorPool,
    command_pool: VkCommandPool,
    memory_properties: VkPhysicalDeviceMemoryProperties,
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    /// Initialize Vulkan, pick the first physical device with a compute
    /// queue, create a logical device + compute queue. Returns `Ok(None)`
    /// if no Vulkan loader / device is present.
    pub fn create() -> Result<Option<Self>, MetalTileError> {
        unsafe {
            // Instance.
            let app_name = CString::new("metaltile").unwrap();
            let app_info = VkApplicationInfo {
                sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pNext: ptr::null(),
                pApplicationName: app_name.as_ptr(),
                applicationVersion: 1,
                pEngineName: app_name.as_ptr(),
                engineVersion: 1,
                // Vulkan 1.2 — matches the shaderc target env we set.
                apiVersion: (1 << 22) | (2 << 12),
            };
            let inst_ci = VkInstanceCreateInfo {
                sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                pApplicationInfo: &app_info,
                enabledLayerCount: 0,
                ppEnabledLayerNames: ptr::null(),
                enabledExtensionCount: 0,
                ppEnabledExtensionNames: ptr::null(),
            };
            let mut instance: VkInstance = ptr::null_mut();
            if vkCreateInstance(&inst_ci, ptr::null(), &mut instance) != VK_SUCCESS {
                return Ok(None);
            }

            // Physical device.
            let mut count: u32 = 0;
            vkEnumeratePhysicalDevices(instance, &mut count, ptr::null_mut());
            if count == 0 {
                vkDestroyInstance(instance, ptr::null());
                return Ok(None);
            }
            let mut phys = vec![ptr::null_mut(); count as usize];
            vk_check(
                vkEnumeratePhysicalDevices(instance, &mut count, phys.as_mut_ptr()),
                "vkEnumeratePhysicalDevices",
            )?;
            let physical_device = phys[0];

            // Pick a queue family supporting compute.
            let mut qcount: u32 = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(
                physical_device,
                &mut qcount,
                ptr::null_mut(),
            );
            let mut qprops: Vec<VkQueueFamilyProperties> = (0..qcount as usize)
                .map(|_| VkQueueFamilyProperties {
                    queueFlags: 0,
                    queueCount: 0,
                    timestampValidBits: 0,
                    minImageTransferGranularity: [0; 3],
                })
                .collect();
            vkGetPhysicalDeviceQueueFamilyProperties(
                physical_device,
                &mut qcount,
                qprops.as_mut_ptr(),
            );
            let queue_family_index = qprops
                .iter()
                .position(|q| q.queueFlags & VK_QUEUE_COMPUTE_BIT != 0)
                .ok_or_else(|| {
                    MetalTileError::Dispatch(
                        "vulkan: no queue family with VK_QUEUE_COMPUTE_BIT".into(),
                    )
                })? as u32;

            // Logical device + queue. Chain Vulkan 1.1 + 1.2 feature
            // structs so we can request `shaderFloat16`, `shaderInt8`,
            // `storageBuffer16BitAccess`, `storageBuffer8BitAccess`, and
            // `scalarBlockLayout` — the bits we need for f16/bf16/i8 SSBO
            // kernels (3538-kernel Vulkan corpus unlock).
            //
            // Both structs are zeroed and then the specific bits set; the
            // driver ignores fields it doesn't support. If the device
            // lacks a feature, vkCreateDevice fails — we fall back to a
            // f32-only device for compatibility on older drivers.
            let prio: f32 = 1.0;
            let queue_ci = VkDeviceQueueCreateInfo {
                sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                queueFamilyIndex: queue_family_index,
                queueCount: 1,
                pQueuePriorities: &prio,
            };
            // Vulkan 1.3: `subgroupSizeControl` lets us pin the compute
            // subgroup size to 32 at pipeline creation — required so
            // metaltile kernels' `subgroupAdd` etc. reduce within a
            // 32-lane SIMD group (matching the Apple/CUDA assumption).
            // Without this, AMD drivers can pick 64 for small workgroups
            // and `simd_sum` silently sums across two simdgroups.
            let mut feat13: VkPhysicalDeviceVulkan13Features = std::mem::zeroed();
            feat13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
            feat13.subgroupSizeControl = VK_TRUE;
            feat13.computeFullSubgroups = VK_TRUE;
            feat13.maintenance4 = VK_TRUE;
            let mut feat12: VkPhysicalDeviceVulkan12Features = std::mem::zeroed();
            feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            feat12.shaderFloat16 = VK_TRUE;
            feat12.shaderInt8 = VK_TRUE;
            feat12.storageBuffer8BitAccess = VK_TRUE;
            feat12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
            feat12.storagePushConstant8 = VK_TRUE;
            feat12.scalarBlockLayout = VK_TRUE;
            feat12.pNext = &mut feat13 as *mut _ as *mut c_void;
            let mut feat11: VkPhysicalDeviceVulkan11Features = std::mem::zeroed();
            feat11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
            feat11.storageBuffer16BitAccess = VK_TRUE;
            feat11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
            feat11.storagePushConstant16 = VK_TRUE;
            feat11.pNext = &mut feat12 as *mut _ as *mut c_void;
            let dev_ci = VkDeviceCreateInfo {
                sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                pNext: &feat11 as *const _ as *const c_void,
                flags: 0,
                queueCreateInfoCount: 1,
                pQueueCreateInfos: &queue_ci,
                enabledLayerCount: 0,
                ppEnabledLayerNames: ptr::null(),
                enabledExtensionCount: 0,
                ppEnabledExtensionNames: ptr::null(),
                pEnabledFeatures: ptr::null(),
            };
            let mut device: VkDevice = ptr::null_mut();
            let create_res = vkCreateDevice(physical_device, &dev_ci, ptr::null(), &mut device);
            if create_res != VK_SUCCESS {
                // Retry without the f16/bf16/i8 chain — old drivers, or
                // devices that don't support these features, get the
                // Phase-1 f32-only path back.
                let plain_ci = VkDeviceCreateInfo {
                    pNext: ptr::null(),
                    ..dev_ci
                };
                vk_check(
                    vkCreateDevice(physical_device, &plain_ci, ptr::null(), &mut device),
                    "vkCreateDevice(plain)",
                )?;
            }
            let mut queue: VkQueue = ptr::null_mut();
            vkGetDeviceQueue(device, queue_family_index, 0, &mut queue);

            // Memory properties (used by every alloc).
            let mut mem_props: VkPhysicalDeviceMemoryProperties = std::mem::zeroed();
            vkGetPhysicalDeviceMemoryProperties(physical_device, &mut mem_props);

            // Descriptor pool: 64 sets × 8 storage buffers each — generous
            // for Phase-1 smoke; real workloads will recycle into a pool
            // sized from the worst-case kernel.
            let pool_sizes = [VkDescriptorPoolSize {
                typ: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 64 * 8,
            }];
            let pool_ci = VkDescriptorPoolCreateInfo {
                sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                maxSets: 64,
                poolSizeCount: 1,
                pPoolSizes: pool_sizes.as_ptr(),
            };
            let mut descriptor_pool: VkDescriptorPool = VK_NULL_HANDLE;
            vk_check(
                vkCreateDescriptorPool(device, &pool_ci, ptr::null(), &mut descriptor_pool),
                "vkCreateDescriptorPool",
            )?;

            // Command pool for transient compute submissions.
            let cmd_ci = VkCommandPoolCreateInfo {
                sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex: queue_family_index,
            };
            let mut command_pool: VkCommandPool = VK_NULL_HANDLE;
            vk_check(
                vkCreateCommandPool(device, &cmd_ci, ptr::null(), &mut command_pool),
                "vkCreateCommandPool",
            )?;

            Ok(Some(VulkanDevice {
                instance,
                physical_device,
                device,
                queue,
                queue_family_index,
                descriptor_pool,
                command_pool,
                memory_properties: mem_props,
            }))
        }
    }

    /// Find a memory type matching `mem_type_bits` (from
    /// `vkGetBufferMemoryRequirements`) with all the requested property flags.
    fn find_memory_type(
        &self,
        mem_type_bits: u32,
        flags: u32,
    ) -> Result<u32, MetalTileError> {
        for i in 0..self.memory_properties.memoryTypeCount {
            if (mem_type_bits & (1u32 << i)) != 0
                && (self.memory_properties.memoryTypes[i as usize].propertyFlags & flags)
                    == flags
            {
                return Ok(i);
            }
        }
        Err(MetalTileError::Dispatch(format!(
            "vulkan: no memory type with flags=0x{flags:x} typeBits=0x{mem_type_bits:x}"
        )))
    }

    /// Allocate a host-visible, host-coherent storage buffer of `size` bytes.
    /// Mapped via `vkMapMemory`; no explicit flush required (`HOST_COHERENT`).
    pub fn alloc_storage(&self, size: u64) -> Result<VulkanBuffer<'_>, MetalTileError> {
        let bci = VkBufferCreateInfo {
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            size,
            usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
        };
        let mut buffer: VkBuffer_ = VK_NULL_HANDLE;
        unsafe {
            vk_check(
                vkCreateBuffer(self.device, &bci, ptr::null(), &mut buffer),
                "vkCreateBuffer",
            )?;
            let mut req = VkMemoryRequirements {
                size: 0,
                alignment: 0,
                memoryTypeBits: 0,
            };
            vkGetBufferMemoryRequirements(self.device, buffer, &mut req);
            let mem_type_index = self.find_memory_type(
                req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )?;
            let ai = VkMemoryAllocateInfo {
                sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                pNext: ptr::null(),
                allocationSize: req.size,
                memoryTypeIndex: mem_type_index,
            };
            let mut memory: VkDeviceMemory = VK_NULL_HANDLE;
            vk_check(
                vkAllocateMemory(self.device, &ai, ptr::null(), &mut memory),
                "vkAllocateMemory",
            )?;
            vk_check(
                vkBindBufferMemory(self.device, buffer, memory, 0),
                "vkBindBufferMemory",
            )?;
            Ok(VulkanBuffer { buffer, memory, size, dev: self })
        }
    }

    /// Allocate + upload host bytes (host-visible buffer, single mapping).
    pub fn upload(&self, data: &[u8]) -> Result<VulkanBuffer<'_>, MetalTileError> {
        let size = data.len().max(4) as u64; // Vulkan rejects 0-byte allocs.
        let buf = self.alloc_storage(size)?;
        if !data.is_empty() {
            unsafe {
                let mut p: *mut c_void = ptr::null_mut();
                vk_check(
                    vkMapMemory(self.device, buf.memory, 0, VK_WHOLE_SIZE, 0, &mut p),
                    "vkMapMemory(upload)",
                )?;
                std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut u8, data.len());
                vkUnmapMemory(self.device, buf.memory);
            }
        }
        Ok(buf)
    }

    /// Read back `out.len()` bytes from a host-visible buffer.
    pub fn download(&self, buf: &VulkanBuffer, out: &mut [u8]) -> Result<(), MetalTileError> {
        if out.is_empty() {
            return Ok(());
        }
        unsafe {
            let mut p: *mut c_void = ptr::null_mut();
            vk_check(
                vkMapMemory(self.device, buf.memory, 0, VK_WHOLE_SIZE, 0, &mut p),
                "vkMapMemory(download)",
            )?;
            std::ptr::copy_nonoverlapping(p as *const u8, out.as_mut_ptr(), out.len());
            vkUnmapMemory(self.device, buf.memory);
        }
        Ok(())
    }

    /// Compile GLSL compute → SPIR-V via shaderc, then build a Vulkan
    /// compute pipeline + descriptor-set layout from it. The `plan` carries
    /// the binding layout the host needs to match the emitter's expectations
    /// (one storage buffer per param, one push-constant block).
    pub fn compile(
        &self,
        glsl_src: &str,
        plan: &GlslBindingPlan,
        kernel_name: &str,
    ) -> Result<VulkanPipeline, MetalTileError> {
        let spv = compile_glsl_to_spv(glsl_src, kernel_name)?;

        // Validate the SPIR-V binary is word-aligned (shaderc returns bytes;
        // VkShaderModuleCreateInfo wants u32*).
        if spv.len() % 4 != 0 {
            return Err(MetalTileError::Compilation(format!(
                "vulkan: SPIR-V size {} not a multiple of 4",
                spv.len()
            )));
        }

        unsafe {
            // Shader module.
            let sm_ci = VkShaderModuleCreateInfo {
                sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                codeSize: spv.len(),
                pCode: spv.as_ptr() as *const u32,
            };
            let mut shader_module: VkShaderModule = VK_NULL_HANDLE;
            vk_check(
                vkCreateShaderModule(self.device, &sm_ci, ptr::null(), &mut shader_module),
                "vkCreateShaderModule",
            )?;

            // Descriptor-set layout: one storage-buffer binding per param.
            let bindings: Vec<VkDescriptorSetLayoutBinding> = plan
                .bindings
                .iter()
                .map(|b| VkDescriptorSetLayoutBinding {
                    binding: b.binding,
                    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount: 1,
                    stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                    pImmutableSamplers: ptr::null(),
                })
                .collect();
            let dsl_ci = VkDescriptorSetLayoutCreateInfo {
                sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                bindingCount: bindings.len() as u32,
                pBindings: bindings.as_ptr(),
            };
            let mut set_layout: VkDescriptorSetLayout = VK_NULL_HANDLE;
            vk_check(
                vkCreateDescriptorSetLayout(
                    self.device,
                    &dsl_ci,
                    ptr::null(),
                    &mut set_layout,
                ),
                "vkCreateDescriptorSetLayout",
            )?;

            // Pipeline layout.
            let pc_range = VkPushConstantRange {
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                offset: 0,
                size: plan.push_constant_bytes.max(4), // Vulkan rejects 0.
            };
            let pl_ci = VkPipelineLayoutCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                setLayoutCount: 1,
                pSetLayouts: &set_layout,
                pushConstantRangeCount: if plan.push_constant_bytes > 0 { 1 } else { 0 },
                pPushConstantRanges: &pc_range,
            };
            let mut layout: VkPipelineLayout = VK_NULL_HANDLE;
            vk_check(
                vkCreatePipelineLayout(self.device, &pl_ci, ptr::null(), &mut layout),
                "vkCreatePipelineLayout",
            )?;

            // Compute pipeline. Pin the subgroup size to 32 (the
            // metaltile kernels' Apple-simdgroup assumption). On AMD
            // RDNA this is wave32; on devices that support
            // `subgroupSizeControl`, this guarantees `subgroupAdd` etc.
            // reduce within a 32-lane SIMD group.
            let req_subgroup = VkPipelineShaderStageRequiredSubgroupSizeCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
                pNext: ptr::null_mut(),
                requiredSubgroupSize: 32,
            };
            let entry =
                CStr::from_bytes_with_nul(ENTRY_POINT).unwrap().as_ptr();
            let stage = VkPipelineShaderStageCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: &req_subgroup as *const _ as *const c_void,
                flags: 0,
                stage: VK_SHADER_STAGE_COMPUTE_BIT,
                module: shader_module,
                pName: entry,
                pSpecializationInfo: ptr::null(),
            };
            let cp_ci = VkComputePipelineCreateInfo {
                sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stage,
                layout,
                basePipelineHandle: VK_NULL_HANDLE,
                basePipelineIndex: -1,
            };
            let mut pipeline: VkPipeline_ = VK_NULL_HANDLE;
            vk_check(
                vkCreateComputePipelines(
                    self.device,
                    VK_NULL_HANDLE,
                    1,
                    &cp_ci,
                    ptr::null(),
                    &mut pipeline,
                ),
                "vkCreateComputePipelines",
            )?;

            Ok(VulkanPipeline {
                pipeline,
                layout,
                set_layout,
                shader_module,
                push_constant_bytes: plan.push_constant_bytes,
            })
        }
    }

    /// End-to-end dispatch: GLSL → SPIR-V → pipeline → bind → dispatch →
    /// readback. Mirrors `HipDevice::run_kernel` / `CudaDevice::run_kernel`
    /// — same 3-D `grid` × `block` calling convention so the corpus
    /// harness can swap devices freely.
    ///
    /// `buffers` maps param names (and constexpr names) to host bytes; we
    /// allocate matching host-visible buffers, upload, dispatch, and
    /// download the output params. The kernel's per-mode bounds guard
    /// (Elementwise's `_n_elems`, Reduction's range loop) handles any
    /// overshoot from grid rounding.
    pub fn run_kernel(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Result<BTreeMap<String, Vec<u8>>, MetalTileError> {
        // 1. Codegen: IR → GLSL + binding plan. Workgroup size matches
        //    the harness's `tpg` so single-warp / multi-warp / 3-D tpg
        //    kernels all map straight through.
        let cg = GlslGenerator::new().with_local_size_3d(block);
        let plan = cg.binding_plan(kernel).map_err(MetalTileError::Codegen)?;
        let glsl = cg.generate(kernel).map_err(MetalTileError::Codegen)?;

        // 2. Compile + build pipeline.
        let pipeline = self.compile(&glsl, &plan, &kernel.name)?;

        // 3. Upload every param as a storage buffer. dev_bufs is kept
        //    in `kernel.params` order; Strided params interleave the
        //    `_shape` and `_strides` companion buffers immediately
        //    after their data, matching the emitter's binding layout.
        let mut dev_bufs: Vec<VulkanBuffer> = Vec::new();
        let mut out_meta: Vec<Option<(String, usize)>> = Vec::new();
        for p in &kernel.params {
            let bytes = buffers.get(&p.name).ok_or_else(|| {
                MetalTileError::Dispatch(format!("missing buffer for param '{}'", p.name))
            })?;
            dev_bufs.push(self.upload(bytes)?);
            out_meta.push(if p.is_output { Some((p.name.clone(), bytes.len())) } else { None });
            if matches!(p.kind, metaltile_core::ir::ParamKind::Strided) {
                for suffix in ["_shape", "_strides"] {
                    let key = format!("{}{}", p.name, suffix);
                    let meta = match buffers.get(&key) {
                        Some(b) => b.clone(),
                        None => synth_strided_meta(&p.shape, suffix == "_strides"),
                    };
                    dev_bufs.push(self.upload(&meta)?);
                    out_meta.push(None);
                }
            }
        }

        // 4. Build push-constant payload (constexprs in order, then `_n_elems`).
        let mut push: Vec<u8> = Vec::with_capacity(plan.push_constant_bytes as usize);
        for ce in &kernel.constexprs {
            let name = ce.name.name();
            let bytes = buffers.get(name).ok_or_else(|| {
                MetalTileError::Dispatch(format!("missing constexpr '{name}'"))
            })?;
            push.extend_from_slice(bytes);
        }
        if plan.has_n_elems {
            let n_elems = kernel
                .params
                .iter()
                .position(|p| p.is_output)
                .and_then(|i| {
                    let p = &kernel.params[i];
                    buffers.get(&p.name).map(|b| (b.len() / p.dtype.size_bytes().max(1)) as u32)
                })
                .unwrap_or(0);
            push.extend_from_slice(&n_elems.to_le_bytes());
        }

        // 5. Allocate descriptor set + update with our buffers.
        let descriptor_set = unsafe {
            let ai = VkDescriptorSetAllocateInfo {
                sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                pNext: ptr::null(),
                descriptorPool: self.descriptor_pool,
                descriptorSetCount: 1,
                pSetLayouts: &pipeline.set_layout,
            };
            let mut ds: VkDescriptorSet = VK_NULL_HANDLE;
            vk_check(
                vkAllocateDescriptorSets(self.device, &ai, &mut ds),
                "vkAllocateDescriptorSets",
            )?;
            ds
        };
        // `buf_infos` MUST outlive the `vkUpdateDescriptorSets` call (it
        // references our slice). Hold it for the whole unsafe block.
        let buf_infos: Vec<VkDescriptorBufferInfo> = dev_bufs
            .iter()
            .map(|b| VkDescriptorBufferInfo {
                buffer: b.buffer,
                offset: 0,
                range: b.size,
            })
            .collect();
        let writes: Vec<VkWriteDescriptorSet> = plan
            .bindings
            .iter()
            .enumerate()
            .map(|(i, b)| VkWriteDescriptorSet {
                sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: ptr::null(),
                dstSet: descriptor_set,
                dstBinding: b.binding,
                dstArrayElement: 0,
                descriptorCount: 1,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo: ptr::null(),
                pBufferInfo: &buf_infos[i],
                pTexelBufferView: ptr::null(),
            })
            .collect();
        unsafe {
            vkUpdateDescriptorSets(
                self.device,
                writes.len() as u32,
                writes.as_ptr(),
                0,
                ptr::null(),
            );
        }

        // 6. Build + submit a command buffer: bind pipeline, push consts,
        //    dispatch, wait.
        unsafe {
            let cb_ai = VkCommandBufferAllocateInfo {
                sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext: ptr::null(),
                commandPool: self.command_pool,
                level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount: 1,
            };
            let mut cb: VkCommandBuffer = ptr::null_mut();
            vk_check(
                vkAllocateCommandBuffers(self.device, &cb_ai, &mut cb),
                "vkAllocateCommandBuffers",
            )?;
            let begin = VkCommandBufferBeginInfo {
                sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext: ptr::null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                pInheritanceInfo: ptr::null(),
            };
            vk_check(vkBeginCommandBuffer(cb, &begin), "vkBeginCommandBuffer")?;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdBindDescriptorSets(
                cb,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.layout,
                0,
                1,
                &descriptor_set,
                0,
                ptr::null(),
            );
            if !push.is_empty() {
                vkCmdPushConstants(
                    cb,
                    pipeline.layout,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    push.len() as u32,
                    push.as_ptr() as *const c_void,
                );
            }
            vkCmdDispatch(cb, grid[0], grid[1], grid[2]);

            // Barrier so the host map sees the compute writes.
            let barrier = VkMemoryBarrier {
                sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                pNext: ptr::null(),
                srcAccessMask: VK_ACCESS_SHADER_WRITE_BIT,
                // Host reads aren't a real Vulkan access stage; the host-
                // coherent property + `vkDeviceWaitIdle` below carry the
                // happens-before. We still flag SHADER_READ for any future
                // dispatch chained from this one.
                dstAccessMask: VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            };
            vkCmdPipelineBarrier(
                cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                1,
                &barrier,
                0,
                ptr::null(),
                0,
                ptr::null(),
            );
            vk_check(vkEndCommandBuffer(cb), "vkEndCommandBuffer")?;

            let submit = VkSubmitInfo {
                sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: 0,
                pWaitSemaphores: ptr::null(),
                pWaitDstStageMask: ptr::null(),
                commandBufferCount: 1,
                pCommandBuffers: &cb,
                signalSemaphoreCount: 0,
                pSignalSemaphores: ptr::null(),
            };
            vk_check(
                vkQueueSubmit(self.queue, 1, &submit, VK_NULL_HANDLE),
                "vkQueueSubmit",
            )?;
            vk_check(vkQueueWaitIdle(self.queue), "vkQueueWaitIdle")?;
            // No vkFreeCommandBuffers wired up — they live with the pool;
            // we destroy the pool at device-drop.
        }

        // 7. Read back outputs.
        let mut out = BTreeMap::new();
        for (buf, meta) in dev_bufs.iter().zip(&out_meta) {
            if let Some((name, len)) = meta {
                let mut host = vec![0u8; *len];
                self.download(buf, &mut host)?;
                out.insert(name.clone(), host);
            }
        }

        // 8. Destroy pipeline objects + reset the descriptor pool so the
        //    next `run_kernel` call doesn't exhaust descriptor slots
        //    (the corpus run quickly hits OUT_OF_POOL_MEMORY after a few
        //    hundred kernels otherwise). Buffers drop with `dev_bufs`.
        unsafe {
            vkDestroyPipeline(self.device, pipeline.pipeline, ptr::null());
            vkDestroyPipelineLayout(self.device, pipeline.layout, ptr::null());
            vkDestroyDescriptorSetLayout(self.device, pipeline.set_layout, ptr::null());
            vkDestroyShaderModule(self.device, pipeline.shader_module, ptr::null());
            vkResetDescriptorPool(self.device, self.descriptor_pool, 0);
        }

        Ok(out)
    }

    /// Query name (best-effort): we don't link the extra "PhysicalDeviceProperties"
    /// FFI in Phase 1, so this is a placeholder.
    pub fn name(&self) -> &str { "vulkan-device" }

    /// Physical handle (for future direct queries via `vkGetPhysicalDeviceProperties`).
    pub fn physical_device_handle(&self) -> VkPhysicalDevice {
        self.physical_device
    }

    /// Queue family index in use.
    pub fn queue_family(&self) -> u32 { self.queue_family_index }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            vkDeviceWaitIdle(self.device);
            if self.command_pool != VK_NULL_HANDLE {
                vkDestroyCommandPool(self.device, self.command_pool, ptr::null());
            }
            if self.descriptor_pool != VK_NULL_HANDLE {
                vkDestroyDescriptorPool(self.device, self.descriptor_pool, ptr::null());
            }
            if !self.device.is_null() {
                vkDestroyDevice(self.device, ptr::null());
            }
            if !self.instance.is_null() {
                vkDestroyInstance(self.instance, ptr::null());
            }
        }
    }
}

/// GLSL → SPIR-V via shaderc. The result is a byte-vector whose length is
/// a multiple of 4 (SPIR-V is a stream of u32 words).
pub fn compile_glsl_to_spv(
    glsl_src: &str,
    file_name: &str,
) -> Result<Vec<u8>, MetalTileError> {
    let csrc =
        CString::new(glsl_src).map_err(|e| MetalTileError::Compilation(e.to_string()))?;
    let cfile =
        CString::new(file_name).map_err(|e| MetalTileError::Compilation(e.to_string()))?;
    let centry = CString::new("main").unwrap();
    unsafe {
        let compiler = shaderc_compiler_initialize();
        if compiler.is_null() {
            return Err(MetalTileError::Compilation(
                "shaderc_compiler_initialize failed".into(),
            ));
        }
        let opts = shaderc_compile_options_initialize();
        shaderc_compile_options_set_target_env(
            opts,
            shaderc_target_env_vulkan,
            shaderc_env_version_vulkan_1_2 as u32,
        );
        // Optimisation level 0: zero-opt, identical to debug. Phase 1
        // prioritises correctness + readable shader disassembly.
        shaderc_compile_options_set_optimization_level(opts, 0);
        let result = shaderc_compile_into_spv(
            compiler,
            csrc.as_ptr(),
            glsl_src.len(),
            shaderc_compute_shader,
            cfile.as_ptr(),
            centry.as_ptr(),
            opts,
        );
        let status = shaderc_result_get_compilation_status(result);
        if status != shaderc_compilation_status_success {
            let msg = shaderc_result_get_error_message(result);
            let m = if msg.is_null() {
                format!("shaderc error code {status}")
            } else {
                CStr::from_ptr(msg).to_string_lossy().into_owned()
            };
            shaderc_result_release(result);
            shaderc_compile_options_release(opts);
            shaderc_compiler_release(compiler);
            return Err(MetalTileError::Compilation(format!(
                "shaderc_compile_into_spv: {m}\n--- glsl ---\n{glsl_src}"
            )));
        }
        let len = shaderc_result_get_length(result);
        let bytes = shaderc_result_get_bytes(result);
        let spv = std::slice::from_raw_parts(bytes, len).to_vec();
        shaderc_result_release(result);
        shaderc_compile_options_release(opts);
        shaderc_compiler_release(compiler);
        Ok(spv)
    }
}

// Suppress unused warnings on DType / c_char if they end up only referenced
// transitively via the FFI types.
#[allow(dead_code)]
fn _force_use_dtype(_d: DType) {}
#[allow(dead_code)]
fn _force_use_c_char(_c: *const c_char) {}
