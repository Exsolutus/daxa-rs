use crate::{command_list::*, context::Context, gpu_resources::*, semaphore::*};

use anyhow::{Context as _, Result};

use ash::{
    Device as LogicalDevice,
    extensions::ext::DebugUtils,
    vk::{self, PhysicalDevice}
};

use gpu_allocator::{vulkan::*, AllocatorDebugSettings, MemoryLocation};

use std::{
    collections::VecDeque,
    ffi::{CStr},
    ops::{Deref},
    mem::{
        ManuallyDrop,
        size_of
    },
    slice,
    sync::{
        Arc,
        atomic::{
            AtomicU64,
            Ordering
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

#[derive(Default)]
pub struct CommandSubmitInfo {
    pub src_stages: PipelineStageFlags,
    pub command_lists: Vec<CommandList>,
    pub wait_binary_semaphores: Vec<BinarySemaphore>,
    pub signal_binary_semaphores: Vec<BinarySemaphore>,
    pub wait_timeline_semaphores: Vec<(TimelineSemaphore, u64)>,
    pub signal_timeline_semaphores: Vec<(TimelineSemaphore, u64)>,
}

pub struct PresentInfo {
    // wait_binary_semaphores: Vec<BinarySemaphore>, // TODO
    // swapchain: Swapchain,
}



pub trait Zombie { }

#[derive(Default)]
pub(crate) struct MainQueueZombies {
    pub command_lists: VecDeque<(u64, CommandListZombie)>,
    pub buffers: VecDeque<(u64, BufferId)>,
    pub images: VecDeque<(u64, ImageId)>,
    pub image_views: VecDeque<(u64, ImageViewId)>,
    pub samplers: VecDeque<(u64, SamplerId)>,
    pub semaphores: VecDeque<(u64, SemaphoreZombie)>,
    // split_barriers: VecDeque<(u64, SplitBarrierZombie)>, // TODO
    // pipelines: VecDeque<(u64, PipelineZombie)>,
    // timeline_query_pools: VecDeque<(u64, TimelineQueryPoolZombie)>,
}

pub(crate) struct DeviceInternal {
    context: Context,
    properties: DeviceProperties,
    info: DeviceInfo,

    physical_device: PhysicalDevice,
    logical_device: LogicalDevice,

    allocator: ManuallyDrop<Allocator>,

    // Main queue
    main_queue: vk::Queue,
    pub main_queue_family: u32,
    pub main_queue_cpu_timeline: AtomicU64,
    main_queue_gpu_timeline_semaphore: vk::Semaphore,

    // GPU resource table
    // gpu_shader_resource_table: GPUShaderResourceTable, // TODO

    null_sampler: vk::Sampler,
    buffer_device_address_buffer: vk::Buffer,
    buffer_device_address_buffer_allocation: Box<Allocation>,

    // Resource recycling
    command_buffer_pool_pool: Mutex<CommandBufferPoolPool>,
    
    pub main_queue_submits: Mutex<VecDeque<(u64, Vec<CommandList>)>>,
    pub main_queue_zombies: Mutex<MainQueueZombies>,
}

#[derive(Clone)]
pub struct Device {
    pub(crate) internal: Arc<DeviceInternal>
}

impl Deref for Device {
    type Target = LogicalDevice;

    fn deref(&self) -> &Self::Target {
        &self.internal.logical_device
    }
}

// Device creation methods
impl Device {
    pub(crate) fn new(
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
        let required_physical_device_features = vk::PhysicalDeviceFeatures::builder()
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
    
        let mut required_physical_device_features_buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
            .buffer_device_address(true)
            .buffer_device_address_capture_replay(true)
            .build();

        let mut required_physical_device_features_descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
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

        let mut required_physical_device_features_host_query_reset = vk::PhysicalDeviceHostQueryResetFeatures::builder()
            .host_query_reset(true)
            .build();

        let mut required_physical_device_features_shader_atomic_int64 = vk::PhysicalDeviceShaderAtomicInt64Features::builder()
            .shader_buffer_int64_atomics(true)
            .shader_shared_int64_atomics(true)
            .build();

        let mut required_physical_device_features_shader_image_atomic_int64 = vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::builder()
            .shader_image_int64_atomics(true)
            .build();

        let mut required_physical_device_features_dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::builder()
            .dynamic_rendering(true)
            .build();

        let mut required_physical_device_features_timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeatures::builder()
            .timeline_semaphore(true)
            .build();

        let mut required_physical_device_features_synchronization_2 = vk::PhysicalDeviceSynchronization2Features::builder()
            .synchronization2(true)
            .build();

        let mut required_physical_device_features_robustness_2 = vk::PhysicalDeviceRobustness2FeaturesEXT::builder()
            .null_descriptor(true)
            .build();

        let mut required_physical_device_features_scalar_layout = vk::PhysicalDeviceScalarBlockLayoutFeatures::builder()
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
            let device_name = format!("{} [Daxa Device]\0", device_info.debug_name);
            let device_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::DEVICE)
                .object_handle(vk::Handle::as_raw(logical_device.handle()))
                .object_name(&CStr::from_ptr(device_name.as_ptr() as *const i8))
                .build();
            context.debug_utils().set_debug_utils_object_name(logical_device.handle(), &device_name_info)?;

            let queue_name = format!("{} [Daxa Device Queue]\0", device_info.debug_name);
            let queue_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::QUEUE)
                .object_handle(vk::Handle::as_raw(main_queue))
                .object_name(&CStr::from_ptr(queue_name.as_ptr() as *const i8))
                .build();
            context.debug_utils().set_debug_utils_object_name(logical_device.handle(), &queue_name_info)?;

            let semaphore_name = format!("{} [Daxa Device TimelineSemaphore]\0", device_info.debug_name);
            let semaphore_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::SEMAPHORE)
                .object_handle(vk::Handle::as_raw(main_queue_gpu_timeline_semaphore))
                .object_name(&CStr::from_ptr(semaphore_name.as_ptr() as *const i8))
                .build();
            context.debug_utils().set_debug_utils_object_name(logical_device.handle(), &semaphore_name_info)?;

            let buffer_name = format!("{} [Daxa Device Buffer Device Address Buffer]\0", device_info.debug_name);
            let buffer_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::BUFFER)
                .object_handle(vk::Handle::as_raw(buffer_device_address_buffer))
                .object_name(&CStr::from_ptr(buffer_name.as_ptr() as *const i8))
                .build();
            context.debug_utils().set_debug_utils_object_name(logical_device.handle(), &buffer_name_info)?;
        }

        Ok(Self {
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

                command_buffer_pool_pool: Default::default(),
            
                main_queue_submits: Default::default(),
                main_queue_zombies: Mutex::new(MainQueueZombies::default()),
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

    pub fn submit_commands(&self, info: CommandSubmitInfo) {
        self.collect_garbage();

        let timeline_value = self.internal.main_queue_cpu_timeline.fetch_add(1, Ordering::AcqRel) + 1;

        let mut submit: (u64, Vec<CommandList>) = (timeline_value, vec![]);

        let mut main_queue_zombies = self.internal.main_queue_zombies.lock().unwrap();
        for command_list in info.command_lists.as_slice() {
            for (id, index) in command_list.internal.deferred_destructions.as_slice() {
                match *index as usize {
                    DEFERRED_DESTRUCTION_BUFFER_INDEX => main_queue_zombies.buffers.push_front((timeline_value, BufferId(id.0))),
                    DEFERRED_DESTRUCTION_IMAGE_INDEX => main_queue_zombies.images.push_front((timeline_value, ImageId(id.0))),
                    DEFERRED_DESTRUCTION_IMAGE_VIEW_INDEX => main_queue_zombies.image_views.push_front((timeline_value, ImageViewId(id.0))),
                    DEFERRED_DESTRUCTION_SAMPLER_INDEX => main_queue_zombies.samplers.push_front((timeline_value, SamplerId(id.0))),
                    _ => ()
                }
            }
        }

        let submit_command_buffers: Vec<vk::CommandBuffer> = info.command_lists.as_slice()
            .iter()
            .map(|command_list| {
                debug_assert!(command_list.internal.recording_complete.load(Ordering::Relaxed), "Command lists must be completed before submission.");
                submit.1.push(command_list.clone());
                command_list.internal.command_buffer
            })
            .collect();

        // Gather semaphores to signal
        // Add main queue timeline signaling as first timeline semaphore signaling
        let mut submit_semaphore_signals = vec![self.internal.main_queue_gpu_timeline_semaphore]; // All timeline semaphores come first, then binary semaphores follow.
        let mut submit_semaphore_signal_values = vec![timeline_value]; // Used for timeline semaphores. Ignored (push dummy value) for binary semaphores.

        for (timeline_semaphore, signal_value) in info.signal_timeline_semaphores {
            submit_semaphore_signals.push(timeline_semaphore.internal.semaphore);
            submit_semaphore_signal_values.push(signal_value);
        }

        for binary_semaphore in info.signal_binary_semaphores {
            submit_semaphore_signals.push(binary_semaphore.internal.semaphore);
            submit_semaphore_signal_values.push(0); // The vulkan spec requires to have dummy values for binary semaphores.
        }

        // Gather semaphores to wait
        // Used to synchronize with previous submits
        let mut submit_semaphore_waits = vec![];
        let mut submit_semaphore_wait_stage_masks = vec![];
        let mut submit_semaphore_wait_values = vec![];

        for (timeline_semaphore, wait_value) in info.wait_timeline_semaphores {
            submit_semaphore_waits.push(timeline_semaphore.internal.semaphore);
            submit_semaphore_wait_stage_masks.push(vk::PipelineStageFlags::ALL_COMMANDS);
            submit_semaphore_wait_values.push(wait_value);
        }

        for binary_semaphore in info.wait_binary_semaphores {
            submit_semaphore_waits.push(binary_semaphore.internal.semaphore);
            submit_semaphore_wait_stage_masks.push(vk::PipelineStageFlags::ALL_COMMANDS);
            submit_semaphore_wait_values.push(0);
        }

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(&submit_semaphore_wait_values)
            .signal_semaphore_values(&submit_semaphore_signal_values)
            .build();

        let submit_info = vk::SubmitInfo::builder()
            .push_next(&mut timeline_info)
            .wait_semaphores(&submit_semaphore_waits)
            .wait_dst_stage_mask(&submit_semaphore_wait_stage_masks)
            .command_buffers(&submit_command_buffers)
            .signal_semaphores(&submit_semaphore_signals)
            .build();
        unsafe { self.queue_submit(self.internal.main_queue, slice::from_ref(&submit_info), vk::Fence::null()).unwrap_unchecked() };
    
        self.internal.main_queue_submits.lock()
            .unwrap()
            .push_front(submit);
    }



    #[inline]
    pub fn collect_garbage(&self) {
        self.internal.main_queue_collect_garbage();
    }



    pub fn create_command_list(&self, info: CommandListInfo) -> Result<CommandList> {
        let (pool, buffer) = self.internal.command_buffer_pool_pool.lock()
            .unwrap()
            .get(self.clone());

        CommandList::new(self.clone(), pool, buffer, info)
    }

    pub fn create_binary_semaphore(&self, info: BinarySemaphoreInfo) -> Result<BinarySemaphore> {
        BinarySemaphore::new(self.clone(), info)
    }

    #[cfg(debug_assertions)]
    #[inline]
    pub(crate) fn debug_utils(&self) -> &DebugUtils {
        &self.internal.context.debug_utils()
    }
}

// Device internal methods
impl DeviceInternal {
    fn main_queue_collect_garbage(&self) {
        let gpu_timeline_value = unsafe { self.logical_device.get_semaphore_counter_value(self.main_queue_gpu_timeline_semaphore) };
        debug_assert_ne!(gpu_timeline_value, Err(vk::Result::ERROR_DEVICE_LOST), "Device lost");
        let gpu_timeline_value = gpu_timeline_value.unwrap();


        fn check_and_cleanup_gpu_resource<T>(zombies: &mut VecDeque<(u64, T)>, cleanup_fn: &dyn Fn(T), gpu_timeline_value: u64) {
            while !zombies.is_empty() {
                let (timeline_value, _) = zombies.back().unwrap();
                
                if *timeline_value > gpu_timeline_value {
                    break;
                }

                let (_, object) = zombies.pop_back().unwrap();
                cleanup_fn(object);
            };
        }


        let submits = &mut self.main_queue_submits.lock().unwrap();
        check_and_cleanup_gpu_resource::<Vec<CommandList>>(
            submits,
            &|_| {},
            gpu_timeline_value
        );

        let mut zombies = self.main_queue_zombies.lock().unwrap();
        check_and_cleanup_gpu_resource::<CommandListZombie>(
            &mut zombies.command_lists,
            &|zombie| {
                self.command_buffer_pool_pool.lock()
                    .unwrap()
                    .put_back((zombie.command_pool, zombie.command_buffer));
            },
            gpu_timeline_value
        );
        // check_and_cleanup_gpu_resource::<BufferId>(
        //     &mut zombies.buffers,
        //     &|id| {
        //         self.cleanup_buffer(id);
        //     },
        //     gpu_timeline_value
        // );
        // check_and_cleanup_gpu_resource::<ImageId>(
        //     &mut zombies.images,
        //     &|id| {
        //         self.cleanup_image(id);
        //     },
        //     gpu_timeline_value
        // );
        // check_and_cleanup_gpu_resource::<ImageViewId>(
        //     &mut zombies.image_views,
        //     &|id| {
        //         self.cleanup_image_view(id);
        //     },
        //     gpu_timeline_value
        // );
        // check_and_cleanup_gpu_resource::<SamplerId>(
        //     &mut zombies.samplers,
        //     &|id| {
        //         self.cleanup_sampler(id);
        //     },
        //     gpu_timeline_value
        // );
        check_and_cleanup_gpu_resource::<SemaphoreZombie>(
            &mut zombies.semaphores,
            &|zombie| {
                unsafe { self.logical_device.destroy_semaphore(zombie.semaphore, None) };
            },
            gpu_timeline_value
        );
        // check_and_cleanup_gpu_resource::<SplitBarrierZombie>(
        //     &mut zombies.split_barriers,
        //     &|zombie| {
        //         unsafe { self.logical_device.destroy_event(zombie.barrier, None) };
        //     },
        //     gpu_timeline_value
        // );
        // check_and_cleanup_gpu_resource::<PipelineZombie>(
        //     &mut zombies.pipelines,
        //     &|zombie| {
        //         unsafe { self.logical_device.destroy_semaphore(zombie.pipeline, None) };
        //     },
        //     gpu_timeline_value
        // );
        // check_and_cleanup_gpu_resource::<PipelineZombie>(
        //     &mut zombies.timeline_query_pools,
        //     &|zombie| {
        //         unsafe { self.logical_device.destroy_query_pool(zombie.query_pool, None) };
        //     },
        //     gpu_timeline_value
        // );
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
            self.main_queue_collect_garbage();

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
        Context::new(ContextInfo::default()).unwrap()
    }

    #[test]
    fn simplest() {
        let daxa_context = context();

        let device = daxa_context.create_device(DeviceInfo::default());

        assert!(device.is_ok())
    }

    #[test]
    fn device_selection() {
        let daxa_context = context();

        // To select a device, you look at its properties and return a score.
        // Daxa will choose the device you scored as the highest.
        let device = daxa_context.create_device(DeviceInfo {
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
        });

        assert!(device.is_ok());

        // Once the device is created, you can query its properties, such
        // as its name and much more! These are the same properties we used
        // to discriminate in the GPU selection.
        unsafe { println!("{:?}", CStr::from_ptr(device.unwrap().properties().device_name.as_ptr())) }
    }
}