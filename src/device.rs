use crate::context::Context;

use anyhow::{Result, Context as _};

use ash::{
    Device as LogicalDevice,
    vk::{
        self,
        PhysicalDevice,
    }
};

use gpu_allocator::{vulkan::*, AllocatorDebugSettings, MemoryLocation};

use std::{
    collections::VecDeque,
    ffi::{
        CStr,
        CString
    },
    ops::Deref,
    mem::{
        ManuallyDrop,
        size_of
    },
    slice,
    sync::{
        Arc,
        atomic::{
            AtomicU64
        },
        Mutex
    },
};

// Re-export
pub use ash::vk::{
    PhysicalDeviceType as DeviceType,
    PhysicalDeviceLimits as DeviceLimits,
    PhysicalDeviceProperties as DeviceProperties,
    PipelineStageFlags,
};



type DeviceSelector = fn(&DeviceProperties) -> i32;

pub fn default_device_selector(device_properties: &DeviceProperties) -> i32 {
    let mut score = 0;

    match device_properties.device_type {
        DeviceType::DISCRETE_GPU => score += 10000,
        DeviceType::VIRTUAL_GPU => score += 1000,
        DeviceType::INTEGRATED_GPU => score += 100,
        _ => ()
    }

    score
}

#[derive(Clone, Copy)]
pub struct DeviceInfo {
    pub selector: DeviceSelector,
    pub debug_name: &'static str,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            selector: default_device_selector,
            debug_name: ""
        }
    }
}

pub struct CommandSubmitInfo {
    src_stage: PipelineStageFlags,
    // commands_lists: Vec<CommandList>, // TODO
    // wait_binary_semaphores: Vec<BinarySemaphore>,
    // signal_binary_semaphores: Vec<BinarySemaphore>,
    // wait_timeline_semaphores: Vec<(TimelineSemaphore, u64)>,
    // signal_timeline_semaphores: Vec<(TimelineSemaphore, u64)>,
}

pub struct PresentInfo {
    // wait_binary_semaphores: Vec<BinarySemaphore>, // TODO
    // swapchain: Swapchain,
}



struct DeviceInternal {
    context: Context,
    properties: DeviceProperties,
    info: DeviceInfo,

    physical_device: PhysicalDevice,
    logical_device: LogicalDevice,

    allocator: ManuallyDrop<Allocator>,

    // Main queue
    main_queue: vk::Queue,
    main_queue_family: u32,
    main_queue_cpu_timeline: AtomicU64,
    main_queue_gpu_timeline_semaphore: vk::Semaphore,

    // GPU resource table
    // gpu_shader_resource_table: GPUShaderResourceTable, // TODO

    null_sampler: vk::Sampler,
    buffer_device_address_buffer: vk::Buffer,
    buffer_device_address_buffer_allocation: Box<Allocation>,

    // Resource recycling
    // #[cfg(threat_safety)] // TODO
    // command_buffer_pool_pool: Mutex<CommandBufferPoolPool>,
    // #[cfg(not(threat_safety))]
    // command_buffer_pool_pool: CommandBufferPoolPool,
    
    // #[cfg(thread_safety)]
    // main_queue_submits_zombies: Mutex<VecDeque<(u64, Vec<CommandList>)>>, // TODO
    // #[cfg(not(thread_safety))]
    // main_queue_submits_zombies: VecDeque<(u64, Vec<CommandList>)>,
    // main_queue_command_list_zombies: VecDeque<(u64, CommandListZombie)>,
    // main_queue_buffer_zombies: VecDeque<(u64, BufferId)>,
    // main_queue_image_zombies: VecDeque<(u64, ImageId)>,
    // main_queue_image_view_zombies: VecDeque<(u64, ImageViewId)>,
    // main_queue_sampler_zombies: VecDeque<(u64, SamplerId)>,
    // main_queue_semaphore_zombies: VecDeque<(u64, SemaphoreZombie)>,
    // main_queue_split_barrier_zombies: VecDeque<(u64, SplitBarrierZombie)>,
    // main_queue_pipeline_zombies: VecDeque<(u64, PipelineZombie)>,
    // main_queue_timeline_query_pool_zombies: VecDeque<(u64, TimelineQueryPoolZombie)>,
}

#[derive(Clone)]
pub struct Device {
    internal: Arc<DeviceInternal>
}

impl Deref for Device {
    type Target = LogicalDevice;

    fn deref(&self) -> &Self::Target {
        &self.internal.logical_device
    }
}

// Device creation methods
impl Device {
    pub fn new(
        device_info: DeviceInfo,
        device_properties: DeviceProperties,
        context: Context,
        physical_device: PhysicalDevice
    ) -> Result<Self> {
        // Select main queue
        let queue_family_properties = unsafe {
            context.get_physical_device_queue_family_properties(physical_device)
        };

        let Some(main_queue_family) = queue_family_properties.iter()
            .enumerate()
            .find_map(|(index, properties)| {
                if properties.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER) {
                    return Some(index as u32)
                }

                None
            }) else {
                #[cfg(debug_assertions)]
                panic!("No suitable queue family found.");
            };

        let queue_priorities = [0.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::builder()
            .flags(vk::DeviceQueueCreateFlags::empty())
            .queue_family_index(main_queue_family)
            .queue_priorities(&queue_priorities)
            .build();

        // Define required device features
        let required_physical_device_features: vk::PhysicalDeviceFeatures = vk::PhysicalDeviceFeatures::builder()
            .image_cube_array(true)
            .multi_draw_indirect(true) // Very useful for GPU driver rendering
            .fill_mode_non_solid(true)
            .wide_lines(true)
            .sampler_anisotropy(true) // Allows for anisotropic filtering
            .fragment_stores_and_atomics(true)
            .shader_storage_image_multisample(true) // Useful for software VRS
            .shader_storage_image_read_without_format(true)
            .shader_storage_image_write_without_format(true)
            .shader_int64(true) // Used for buffer device address math
            .build();
    
        let mut required_physical_device_features_buffer_device_address: vk::PhysicalDeviceBufferDeviceAddressFeatures = vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
            .buffer_device_address(true)
            .buffer_device_address_capture_replay(true)
            .build();

        let mut required_physical_device_features_descriptor_indexing: vk::PhysicalDeviceDescriptorIndexingFeatures = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
            .shader_sampled_image_array_non_uniform_indexing(true) // Needed for bindless sampled images
            .shader_storage_image_array_non_uniform_indexing(true) // Needed for bindless storage images
            .shader_storage_buffer_array_non_uniform_indexing(true) // Needed for bindless buffers
            .descriptor_binding_sampled_image_update_after_bind(true) // Needed for bindless sampled images
            .descriptor_binding_storage_image_update_after_bind(true) // Needed for bindless storage images
            .descriptor_binding_storage_buffer_update_after_bind(true) // Needed for bindless buffers
            .descriptor_binding_update_unused_while_pending(true) // Needed for bindless table updates
            .descriptor_binding_partially_bound(true) // Needed for sparse binding in bindless table
            .runtime_descriptor_array(true) // Allows bindless table without hardcoded descriptor maximum in shaders
            .build();

        let mut required_physical_device_features_host_query_reset: vk::PhysicalDeviceHostQueryResetFeatures = vk::PhysicalDeviceHostQueryResetFeatures::builder()
            .host_query_reset(true)
            .build();

        let mut required_physical_device_features_shader_atomic_int64: vk::PhysicalDeviceShaderAtomicInt64Features = vk::PhysicalDeviceShaderAtomicInt64Features::builder()
            .shader_buffer_int64_atomics(true)
            .shader_shared_int64_atomics(true)
            .build();

        let mut required_physical_device_features_shader_image_atomic_int64: vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT = vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::builder()
            .shader_image_int64_atomics(true)
            .build();

        let mut required_physical_device_features_dynamic_rendering: vk::PhysicalDeviceDynamicRenderingFeatures = vk::PhysicalDeviceDynamicRenderingFeatures::builder()
            .dynamic_rendering(true)
            .build();

        let mut required_physical_device_features_timeline_semaphore: vk::PhysicalDeviceTimelineSemaphoreFeatures = vk::PhysicalDeviceTimelineSemaphoreFeatures::builder()
            .timeline_semaphore(true)
            .build();

        let mut required_physical_device_features_synchronization_2: vk::PhysicalDeviceSynchronization2Features = vk::PhysicalDeviceSynchronization2Features::builder()
            .synchronization2(true)
            .build();

        let mut required_physical_device_features_robustness_2: vk::PhysicalDeviceRobustness2FeaturesEXT = vk::PhysicalDeviceRobustness2FeaturesEXT::builder()
            .null_descriptor(true)
            .build();

        let mut required_physical_device_features_scalar_layout: vk::PhysicalDeviceScalarBlockLayoutFeatures = vk::PhysicalDeviceScalarBlockLayoutFeatures::builder()
            .scalar_block_layout(true)
            .build();

        let mut physical_device_features_2 = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut required_physical_device_features_buffer_device_address)
            .push_next(&mut required_physical_device_features_descriptor_indexing)
            .push_next(&mut required_physical_device_features_host_query_reset)
            .push_next(&mut required_physical_device_features_shader_atomic_int64)
            .push_next(&mut required_physical_device_features_shader_image_atomic_int64)
            .push_next(&mut required_physical_device_features_dynamic_rendering)
            .push_next(&mut required_physical_device_features_timeline_semaphore)
            .push_next(&mut required_physical_device_features_synchronization_2)
            .push_next(&mut required_physical_device_features_robustness_2)
            .push_next(&mut required_physical_device_features_scalar_layout)
            .features(required_physical_device_features)
            .build();

        // Define device extensions to request
        let extension_names = [
            ash::extensions::khr::Swapchain::name().as_ptr(),
            vk::ExtDescriptorIndexingFn::name().as_ptr(),
            vk::ExtShaderImageAtomicInt64Fn::name().as_ptr(),
            vk::ExtMultiDrawFn::name().as_ptr(),
            #[cfg(conservative_rasterization)]
            vk::ExtConservativeRasterizationFn::name().as_ptr()
        ];

        // Create logical device
        let device_ci = vk::DeviceCreateInfo::builder()
            .push_next(&mut physical_device_features_2)
            .flags(vk::DeviceCreateFlags::empty())
            .queue_create_infos(slice::from_ref(&queue_ci))
            .enabled_extension_names(&extension_names);
        let logical_device = unsafe {
            (*context).create_device(physical_device, &device_ci, None)
                .context("Logical device should be created.")?
        };

        // Create allocator
        let mut allocator = Allocator::new(
            &AllocatorCreateDesc {
                instance: (*context).clone(),
                device: logical_device.clone(),
                physical_device,
                debug_settings: AllocatorDebugSettings {
                    log_memory_information: false,
                    log_leaks_on_shutdown: true,
                    store_stack_traces: false,
                    log_allocations: true,
                    log_frees: true,
                    log_stack_traces: false
                },
                buffer_device_address: true
            }
        ).context("Allocator should be created.")?;

        // Create main queue
        let main_queue = unsafe {
            logical_device.get_device_queue(main_queue_family, 0)
        };
        let main_queue_gpu_timeline_semaphore = unsafe {
            logical_device.create_semaphore(
                &vk::SemaphoreCreateInfo::builder()
                    .push_next(&mut vk::SemaphoreTypeCreateInfo::builder()
                        .semaphore_type(vk::SemaphoreType::TIMELINE)
                        .initial_value(0)
                    )
                    .build(),
                None
            ).context("Device should create a semaphore.")?
        };

        // resources
        let max_buffers = device_properties.limits.max_descriptor_set_storage_buffers.min(100000);

        // Images and buffers can be set to be a null descriptor.
        // Null descriptors are not available for samples, but we still want to have one to overwrite dead resources with.
        // So we create a default sampler that acts as the "null sampler".
        let sampler_ci = vk::SamplerCreateInfo::default();
        let null_sampler = unsafe {
            logical_device.create_sampler(&sampler_ci, None)
                .context("Device should create a sampler.")?
        };

        let buffer_ci = vk::BufferCreateInfo::builder()
            .size((max_buffers * size_of::<u64>() as u32) as vk::DeviceSize)
            .usage(
                vk::BufferUsageFlags::TRANSFER_SRC |
                vk::BufferUsageFlags::TRANSFER_DST |
                vk::BufferUsageFlags::STORAGE_BUFFER
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(slice::from_ref(&main_queue_family))
            .build();

        let buffer_device_address_buffer = unsafe {
            logical_device.create_buffer(&buffer_ci, None)
                .context("Device should create a buffer.")?
        };

        let requirements = unsafe { logical_device.get_buffer_memory_requirements(buffer_device_address_buffer) };
        let buffer_device_address_buffer_allocation = allocator.allocate(
            &AllocationCreateDesc {
                name: format!("{} [Daxa Device Buffer Device Address Buffer]", device_info.debug_name).as_str(),
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true
            }
        ).context("Allocator should allocate memory for buffer.")?;

        unsafe {
            logical_device.bind_buffer_memory(
                buffer_device_address_buffer,
                buffer_device_address_buffer_allocation.memory(),
                buffer_device_address_buffer_allocation.offset()
            ).context("Device should bind buffer memory.")?
        };

        #[cfg(debug_assertions)]
        unsafe {
            let device_name = format!("{} [Daxa Device]", device_info.debug_name);
            let device_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::DEVICE)
                .object_handle(vk::Handle::as_raw(logical_device.handle()))
                .object_name(&CStr::from_ptr(device_name.as_ptr() as *const i8))
                .build();
            context.debug_utils().set_debug_utils_object_name(logical_device.handle(), &device_name_info)?;

            let queue_name = format!("{} [Daxa Device Queue]", device_info.debug_name);
            let queue_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::QUEUE)
                .object_handle(vk::Handle::as_raw(main_queue))
                .object_name(&CStr::from_ptr(queue_name.as_ptr() as *const i8))
                .build();
            context.debug_utils().set_debug_utils_object_name(logical_device.handle(), &queue_name_info)?;

            let semaphore_name = format!("{} [Daxa Device TimelineSemaphore]", device_info.debug_name);
            let semaphore_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::SEMAPHORE)
                .object_handle(vk::Handle::as_raw(main_queue_gpu_timeline_semaphore))
                .object_name(&CStr::from_ptr(semaphore_name.as_ptr() as *const i8))
                .build();
            context.debug_utils().set_debug_utils_object_name(logical_device.handle(), &semaphore_name_info)?;

            let buffer_name = format!("{} [Daxa Device Buffer Device Address Buffer]", device_info.debug_name);
            let buffer_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::BUFFER)
                .object_handle(vk::Handle::as_raw(buffer_device_address_buffer))
                .object_name(&CStr::from_ptr(buffer_name.as_ptr() as *const i8))
                .build();
            context.debug_utils().set_debug_utils_object_name(logical_device.handle(), &buffer_name_info)?;
        }

        Ok(Device {
            internal: Arc::new(DeviceInternal {
                context,
                properties: device_properties,
                info: device_info,
            
                physical_device,
                logical_device,
            
                allocator: ManuallyDrop::new(allocator),
            
                // Main queue
                main_queue,
                main_queue_family,
                main_queue_cpu_timeline: AtomicU64::default(),
                main_queue_gpu_timeline_semaphore,
            
                // GPU resource table
                // gpu_shader_resource_table: GPUShaderResourceTable,
            
                null_sampler,
                buffer_device_address_buffer,
                buffer_device_address_buffer_allocation: Box::new(buffer_device_address_buffer_allocation),
            })
        })
    }
}

// Device usage methods
impl Device {
    #[inline]
    pub fn info(&self) -> &DeviceInfo {
        &self.internal.info
    }

    #[inline]
    pub fn properties(&self) -> DeviceProperties {
        self.internal.properties
    }

    #[inline]
    pub fn wait_idle(&self) {
        self.internal.wait_idle();
    }

    #[inline]
    pub fn collect_garbage(&self) {
        self.internal.main_queue_collect_garbage();
    }
}

// Device internal methods
impl DeviceInternal {
    fn main_queue_collect_garbage(&self) {
        // TODO
        todo!()
    }

    fn wait_idle(&self) {
        unsafe {
            // `unwrap_unchecked` is safe because these Results are unused
            // TODO: handle relevant Vulkan return codes elsewhere
            self.logical_device.queue_wait_idle(self.main_queue).unwrap_unchecked();
            self.logical_device.device_wait_idle().unwrap_unchecked();
        }
    }
}

impl Drop for DeviceInternal {
    fn drop(&mut self) {
        unsafe {
            self.wait_idle();
            //self.main_queue_collect_garbage();

            //self.allocator.free(self.buffer_device_address_buffer_allocation.take());
            self.logical_device.destroy_buffer(self.buffer_device_address_buffer, None);

            ManuallyDrop::drop(&mut self.allocator);
            self.logical_device.destroy_sampler(self.null_sampler, None);
            self.logical_device.destroy_semaphore(self.main_queue_gpu_timeline_semaphore, None);
            self.logical_device.destroy_device(None);
        }
    }
}



#[cfg(test)]
mod tests {
    use crate::context::{Context, ContextInfo};
    use super::{DeviceInfo, DeviceType};
    use std::ffi::CStr;

    fn context() -> Context {
        Context::new(ContextInfo::default())
            .expect("Context should be created.")
    }

    #[test]
    fn simplest() {
        let _daxa_context = context();

        let _device = _daxa_context.create_device(DeviceInfo::default());
    }

    #[test]
    fn device_selection() {
        let _daxa_context = context();

        // To select a device, you look at its properties and return a score.
        // Daxa will choose the device you scored as the highest.
        let _device = _daxa_context.create_device(DeviceInfo {
            selector: |&properties| {
                let mut score = 0;

                match properties.device_type {
                    DeviceType::DISCRETE_GPU => score += 10000,
                    DeviceType::VIRTUAL_GPU => score += 1000,
                    DeviceType::INTEGRATED_GPU => score += 100,
                    _ => ()
                }

                score
            },
            debug_name: "My device",
        }).expect("Device should be created.");

        unsafe {
            println!("{:?}", CStr::from_ptr(_device.properties().device_name.as_ptr()))
        }
        
    }
}