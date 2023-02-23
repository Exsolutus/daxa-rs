mod info;
pub use info::*;

use crate::{
    core::*,
    device::Device,
    gpu_resources::GPUResourceId,
};
use anyhow::{Context as _, Result};
use ash::vk;
use std::{
    ffi::{
        CStr
    },
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            AtomicUsize,
            Ordering,
        }
    }
};



pub const DEFERRED_DESTRUCTION_BUFFER_INDEX: usize = 0;
pub const DEFERRED_DESTRUCTION_IMAGE_INDEX: usize = 1;
pub const DEFERRED_DESTRUCTION_IMAGE_VIEW_INDEX: usize = 2;
pub const DEFERRED_DESTRUCTION_SAMPLER_INDEX: usize = 3;
pub const DEFERRED_DESTRUCTION_TIMELINE_QUERY_POOL_INDEX: usize = 4;
pub const DEFERRED_DESTRUCTION_COUNT_MAX: usize = 32;

pub const COMMAND_LIST_BARRIER_MAX_BATCH_SIZE: usize = 16;
pub const COMMAND_LIST_COLOR_ATTACHMENT_MAX: usize = 16;



#[derive(Default)]
pub struct CommandBufferPoolPool {
    pools_and_buffers: Vec<(vk::CommandPool, vk::CommandBuffer)>
}

impl CommandBufferPoolPool {
    pub(crate) fn get(&mut self, device: Device) -> (vk::CommandPool, vk::CommandBuffer) {
        self.pools_and_buffers.pop()
            .or_else(|| {
                let command_pool_ci = vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(device.internal.main_queue_family);

                let command_pool = unsafe {
                    device.create_command_pool(&command_pool_ci, None).unwrap_unchecked()
                };

                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                let command_buffer = unsafe {
                    device.allocate_command_buffers(&command_buffer_allocate_info).unwrap_unchecked()[0]
                };

                Some((command_pool, command_buffer))
            })
            .unwrap()
    }

    pub(crate) fn put_back(&mut self, pool_and_buffer: (vk::CommandPool, vk::CommandBuffer)) {
        self.pools_and_buffers.push(pool_and_buffer);
    }

    pub(crate) fn cleanup(&mut self, device: Device) {
        for (pool, _) in &self.pools_and_buffers {
            unsafe { device.destroy_command_pool(*pool, None) };
        }
        self.pools_and_buffers.clear();
    }
}



pub struct CommandListZombie {
    pub command_buffer: vk::CommandBuffer,
    pub command_pool: vk::CommandPool,
}

pub(crate) struct CommandListInternal {
    device: Device,
    info: CommandListInfo,
    pub command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    pub recording_complete: AtomicBool,
    memory_barrier_batch: [vk::MemoryBarrier2; COMMAND_LIST_BARRIER_MAX_BATCH_SIZE],
    image_barrier_batch: [vk::ImageMemoryBarrier2; COMMAND_LIST_BARRIER_MAX_BATCH_SIZE],
    memory_barrier_batch_count: AtomicUsize,
    image_barrier_batch_count: AtomicUsize,
    split_barrier_batch_count: AtomicUsize,
    // pipeline_layouts: [vk::PipelineLayout; PIPELINE_LAYOUT_COUNT as usize],
    pub deferred_destructions: Vec<(GPUResourceId, u8)>
}

#[derive(Clone)]
pub struct CommandList {
    pub(crate) internal: Arc<CommandListInternal>
}

// CommandList creation methods
impl CommandList {
    pub fn new(
        device: Device,
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        info: CommandListInfo
    ) -> Result<Self> {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(command_buffer, &begin_info).unwrap_unchecked() };
        
        #[cfg(debug_assertions)]
        unsafe {
            let command_buffer_name = format!("{} [Daxa CommandBuffer]\0", info.debug_name);
            let command_buffer_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::COMMAND_BUFFER)
                .object_handle(vk::Handle::as_raw(command_buffer))
                .object_name(&CStr::from_ptr(command_buffer_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.handle(), &command_buffer_name_info)?;

            let command_pool_name = format!("{} [Daxa CommandPool]\0", info.debug_name);
            let command_pool_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::COMMAND_POOL)
                .object_handle(vk::Handle::as_raw(command_pool))
                .object_name(&CStr::from_ptr(command_pool_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.handle(), &command_pool_name_info)?;
        }

        Ok(Self {
            internal: Arc::new(CommandListInternal {
                device,
                info,
                command_buffer,
                command_pool,
                recording_complete: AtomicBool::new(false),
                memory_barrier_batch: Default::default(),
                image_barrier_batch: Default::default(),
                memory_barrier_batch_count: AtomicUsize::new(0),
                image_barrier_batch_count: AtomicUsize::new(0),
                split_barrier_batch_count: AtomicUsize::new(0),
                // pipeline_layouts: Default::default(),
                deferred_destructions: vec![]
            })
        })
    }
}

// CommandList usage methods
impl CommandList {
    pub fn complete(&self) {
        debug_assert_eq!(
            self.internal.recording_complete.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed), 
            Ok(false),
            "CommandList is already completed."
        );

        self.internal.flush_barriers();

        unsafe { self.internal.device.end_command_buffer(self.internal.command_buffer).unwrap_unchecked() };
    }
}

// CommandList internal methods
impl CommandListInternal {
    fn flush_barriers(&self) {
        let memory_barrier_batch_count = self.memory_barrier_batch_count.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |_| Some(0)).unwrap(); 
        let image_barrier_batch_count = self.image_barrier_batch_count.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |_| Some(0)).unwrap();

        if memory_barrier_batch_count == 0 && image_barrier_batch_count == 0 {
            return;
        }

        let dependency_info = vk::DependencyInfo::builder()
            .dependency_flags(vk::DependencyFlags::empty())
            .memory_barriers(&self.memory_barrier_batch[0..memory_barrier_batch_count])
            .image_memory_barriers(&self.image_barrier_batch[0..image_barrier_batch_count]);

        unsafe {
            self.device.cmd_pipeline_barrier2(self.command_buffer, &dependency_info);    
        }
    }
}

impl Drop for CommandListInternal {
    fn drop(&mut self) {
        unsafe {
            self.device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty()).unwrap_unchecked();

            let main_queue_zombies = &mut self.device.internal.main_queue_zombies.lock().unwrap();

            let main_queue_cpu_timeline = self.device.internal.main_queue_cpu_timeline.load(Ordering::Acquire);

            main_queue_zombies.command_lists.push_front((
                main_queue_cpu_timeline,
                CommandListZombie {
                    command_buffer: self.command_buffer,
                    command_pool: self.command_pool
                }
            ))
        }
    }
}



#[cfg(test)]
mod tests {
    use crate::{context::*, device::*, gpu_resources::*};
    use super::{CommandList, CommandListInfo};
    use ash::vk;

    struct App {
        daxa_context: Context,
        device: Device
    }

    impl App {
        fn new() -> App {
            let daxa_context = Context::new(ContextInfo::default()).unwrap();
            let device = daxa_context.create_device(DeviceInfo::default()).unwrap();

            App {
                daxa_context,
                device
            }
        }
    }

    #[test]
    fn simplest() {
        let app = App::new();

        let command_list = app.device.create_command_list(CommandListInfo::default()).unwrap();

        // Command lists must be completed before submission!
        command_list.complete();

        app.device.submit_commands(CommandSubmitInfo {
            command_lists: vec![command_list],
            ..Default::default()
        })
    }

    #[test]
    fn deferred_destruction() {
        let app = App::new();

        let command_list = app.device.create_command_list(CommandListInfo {
            debug_name: "deferred_destruction command list"
        });

        // let buffer = app.device.create_buffer(BufferInfo {
        //     size: 4,
        //     .
        // })

    }
}