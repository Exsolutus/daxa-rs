mod info;
pub use info::*;

use crate::{
    device::{Device, DeviceInternal},
    gpu_resources::{
        GPUResourceId,
        BufferId,
        ImageId,
        ImageViewId,
        SamplerId
    },
    split_barrier::*
};
use anyhow::{Result, bail};
use ash::vk;
use std::{
    ffi::{
        CStr
    },
    slice,
    sync::Arc
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
pub(crate) struct CommandBufferPoolPool {
    pools_and_buffers: Vec<(vk::CommandPool, vk::CommandBuffer)>
}

impl CommandBufferPoolPool {
    pub fn get(&mut self, device: &DeviceInternal) -> (vk::CommandPool, vk::CommandBuffer) {
        self.pools_and_buffers.pop()
            .or_else(|| {
                let command_pool_ci = vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(device.main_queue_family);

                let command_pool = unsafe {
                    device.logical_device.create_command_pool(&command_pool_ci, None).unwrap_unchecked()
                };

                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                let command_buffer = unsafe {
                    device.logical_device.allocate_command_buffers(&command_buffer_allocate_info).unwrap_unchecked()[0]
                };

                Some((command_pool, command_buffer))
            })
            .unwrap()
    }

    pub fn put_back(&mut self, pool_and_buffer: (vk::CommandPool, vk::CommandBuffer)) {
        self.pools_and_buffers.push(pool_and_buffer);
    }

    pub fn cleanup(&mut self, device: &DeviceInternal) {
        for (pool, _) in &self.pools_and_buffers {
            unsafe { device.logical_device.destroy_command_pool(*pool, None) };
        }
        self.pools_and_buffers.clear();
    }
}


pub(crate) struct CommandListZombie {
    pub command_buffer: vk::CommandBuffer,
    pub command_pool: vk::CommandPool,
}


pub struct CommandList(pub(crate) CommandListState);

pub(crate) enum CommandListState {
    Recording(Box<CommandListInternal>),
    Completed(Arc<CommandListInternal>)  
}

pub(crate) struct CommandListInternal {
    device: Device,
    info: CommandListInfo,
    pub command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    memory_barrier_batch: [vk::MemoryBarrier2; COMMAND_LIST_BARRIER_MAX_BATCH_SIZE],
    image_barrier_batch: [vk::ImageMemoryBarrier2; COMMAND_LIST_BARRIER_MAX_BATCH_SIZE],
    memory_barrier_batch_count: usize,
    image_barrier_batch_count: usize,
    split_barrier_batch_count: usize,
    // pipeline_layouts: [vk::PipelineLayout; PIPELINE_LAYOUT_COUNT as usize],
    pub deferred_destructions: Vec<(GPUResourceId, u8)>
}

// CommandList creation methods
impl CommandList {
    pub(crate) fn new(
        device: Device,
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        info: CommandListInfo
    ) -> Result<Self> {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.0.logical_device.begin_command_buffer(command_buffer, &begin_info).unwrap_unchecked() };
        
        #[cfg(debug_assertions)]
        unsafe {
            let command_buffer_name = format!("{} [Daxa CommandBuffer]\0", info.debug_name);
            let command_buffer_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::COMMAND_BUFFER)
                .object_handle(vk::Handle::as_raw(command_buffer))
                .object_name(&CStr::from_ptr(command_buffer_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &command_buffer_name_info)?;

            let command_pool_name = format!("{} [Daxa CommandPool]\0", info.debug_name);
            let command_pool_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::COMMAND_POOL)
                .object_handle(vk::Handle::as_raw(command_pool))
                .object_name(&CStr::from_ptr(command_pool_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &command_pool_name_info)?;
        }

        Ok(Self(CommandListState::Recording(Box::new(CommandListInternal {
            device,
            info,
            command_buffer,
            command_pool,
            memory_barrier_batch: Default::default(),
            image_barrier_batch: Default::default(),
            memory_barrier_batch_count: 0,
            image_barrier_batch_count: 0,
            split_barrier_batch_count: 0,
            // pipeline_layouts: Default::default(),
            deferred_destructions: vec![]
        }))))
    }
}

impl Clone for CommandList {
    fn clone(&self) -> Self {
        match &self.0 {
            CommandListState::Recording(_) => {
                panic!("Cannot clone reference to incomplete command list.")
            },
            CommandListState::Completed(internal) => {
                CommandList(CommandListState::Completed(internal.clone()))
            }
        }
    }
}

// CommandList usage methods
impl CommandList {
    pub fn copy_buffer_to_buffer(&mut self, info: BufferCopyInfo) {
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                internal.flush_barriers();

                let buffer_copy = vk::BufferCopy::builder()
                    .src_offset(info.src_offset)
                    .dst_offset(info.dst_offset)
                    .size(info.size);

                unsafe { 
                    internal.device.0.logical_device.cmd_copy_buffer(
                        internal.command_buffer,
                        internal.device.0.buffer_slot(info.src_buffer).buffer,
                        internal.device.0.buffer_slot(info.dst_buffer).buffer,
                        slice::from_ref(&buffer_copy)
                    );
                }
            },
            CommandListState::Completed(_) => panic!("CommandList is already completed.")
        }
    }

    pub fn copy_buffer_to_image(&mut self, info: BufferImageCopyInfo) {
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                internal.flush_barriers();

                let buffer_image_copy = vk::BufferImageCopy::builder()
                    .buffer_offset(info.buffer_offset)
                    .image_subresource(info.image_layers)
                    .image_offset(info.image_offset)
                    .image_extent(info.image_extent);

                unsafe { 
                    internal.device.0.logical_device.cmd_copy_buffer_to_image(
                        internal.command_buffer,
                        internal.device.0.buffer_slot(info.buffer).buffer,
                        internal.device.0.image_slot(info.image).image,
                        info.image_layout,
                        slice::from_ref(&buffer_image_copy)
                    );
                }
            },
            CommandListState::Completed(_) => panic!("CommandList is already completed.")
        }
    }

    pub fn copy_image_to_buffer(&mut self, info: ImageBufferCopyInfo) {
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                internal.flush_barriers();

                let buffer_image_copy = vk::BufferImageCopy::builder()
                    .buffer_offset(info.buffer_offset)
                    .image_subresource(info.image_layers)
                    .image_offset(info.image_offset)
                    .image_extent(info.image_extent);

                unsafe { 
                    internal.device.0.logical_device.cmd_copy_image_to_buffer(
                        internal.command_buffer,
                        internal.device.0.image_slot(info.image).image,
                        info.image_layout,
                        internal.device.0.buffer_slot(info.buffer).buffer,
                        slice::from_ref(&buffer_image_copy)
                    );
                }
            },
            CommandListState::Completed(_) => panic!("CommandList is already completed.")
        }
    }

    pub fn copy_image_to_image(&mut self, info: ImageCopyInfo) {
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                internal.flush_barriers();

                let image_copy = vk::ImageCopy::builder()
                    .src_subresource(info.src_layers)
                    .src_offset(info.src_offset)
                    .dst_subresource(info.dst_layers)
                    .dst_offset(info.dst_offset)
                    .extent(info.extent);

                unsafe { 
                    internal.device.0.logical_device.cmd_copy_image(
                        internal.command_buffer,
                        internal.device.0.image_slot(info.src_image).image,
                        info.src_image_layout,
                        internal.device.0.image_slot(info.dst_image).image,
                        info.dst_image_layout,
                        slice::from_ref(&image_copy)
                    )
                }
            },
            CommandListState::Completed(_) => panic!("CommandList is already completed.")
        }
    }

    pub fn blit_image_to_image(&mut self, info: ImageBlitInfo) {
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                internal.flush_barriers();

                let image_blit = vk::ImageBlit::builder()
                    .src_subresource(info.src_layers)
                    .src_offsets(info.src_offsets)
                    .dst_subresource(info.dst_layers)
                    .dst_offsets(info.dst_offsets);

                unsafe { 
                    internal.device.0.logical_device.cmd_blit_image(
                        internal.command_buffer,
                        internal.device.0.image_slot(info.src_image).image,
                        info.src_image_layout,
                        internal.device.0.image_slot(info.dst_image).image,
                        info.dst_image_layout,
                        slice::from_ref(&image_blit),
                        info.filter
                    )
                }
            },
            CommandListState::Completed(_) => panic!("CommandList is already completed.")
        }
    }


    pub fn clear_buffer(&mut self, info: BufferClearInfo) {
        let CommandListState::Recording(command_list) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
        };

        command_list.flush_barriers();

        unsafe {
            let device = &command_list.device.0;
            device.logical_device.cmd_fill_buffer(
                command_list.command_buffer,
                device.buffer_slot(info.buffer).buffer,
                info.offset,
                info.size,
                info.clear_value
            )
        }
    }

    pub fn clear_image(&mut self, info: &ImageClearInfo) {
        let CommandListState::Recording(command_list) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
        };

        command_list.flush_barriers();

        if info.image_range.aspect_mask.contains(vk::ImageAspectFlags::COLOR) {
            unsafe {
                command_list.device.0.logical_device.cmd_clear_color_image(
                    command_list.command_buffer,
                    command_list.device.0.image_slot(info.image).image,
                    info.image_layout,
                    &info.clear_value.color,
                    slice::from_ref(&info.image_range)
                )
            }
        }

        if info.image_range.aspect_mask.contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL) {
            unsafe {
                command_list.device.0.logical_device.cmd_clear_depth_stencil_image(
                    command_list.command_buffer,
                    command_list.device.0.image_slot(info.image).image,
                    info.image_layout,
                    &info.clear_value.depth_stencil,
                    slice::from_ref(&info.image_range)
                )
            }
        }
    }


    pub fn pipeline_barrier(&mut self, info: MemoryBarrierInfo) {
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                if internal.memory_barrier_batch_count == COMMAND_LIST_BARRIER_MAX_BATCH_SIZE {
                    internal.flush_barriers();
                }

                internal.memory_barrier_batch[internal.memory_barrier_batch_count] = vk::MemoryBarrier2::builder()
                    .src_stage_mask(info.awaited_pipeline_access.0)
                    .src_access_mask(info.awaited_pipeline_access.1)
                    .dst_stage_mask(info.waiting_pipeline_access.0)
                    .dst_access_mask(info.waiting_pipeline_access.1)
                    .build();
            },
            CommandListState::Completed(_) => panic!("CommandList is already completed.")
        }
    }

    pub fn pipeline_barrier_image_transition(&mut self, info: ImageBarrierInfo) {
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                if internal.image_barrier_batch_count == COMMAND_LIST_BARRIER_MAX_BATCH_SIZE {
                    internal.flush_barriers();
                }

                internal.image_barrier_batch[internal.image_barrier_batch_count] = vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(info.awaited_pipeline_access.0)
                    .src_access_mask(info.awaited_pipeline_access.1)
                    .dst_stage_mask(info.waiting_pipeline_access.0)
                    .dst_access_mask(info.waiting_pipeline_access.1)
                    .old_layout(info.before_layout)
                    .new_layout(info.after_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(internal.device.0.image_slot(info.image).image)
                    .subresource_range(info.range)
                    .build();
                internal.image_barrier_batch_count += 1;
            },
            CommandListState::Completed(_) => panic!("CommandList is already completed.")
        }
    }

    pub fn signal_split_barrier(&self, info: SplitBarrierSignalInfo) {
        todo!()
    }

    pub fn wait_split_barriers(&self, infos: &[SplitBarrierWaitInfo]) {
        todo!()
    }

    pub fn wait_split_barrier(&self, info: SplitBarrierWaitInfo) {
        todo!()
    }

    pub fn reset_split_barrier(&self, info: ResetSplitBarrierInfo) {
        todo!()
    }


    // pub fn push_constant(&self, data: *const c_void, size: u32, offset: u32) {
    //     todo!()
    // }

    pub fn push_constant<T>(&self, data: &[T], offset: u32) {
        todo!()
    }

    // pub fn set_pipeline(&self, pipeline: ComputePipeline) {
    //     todo!()
    // }

    // pub fn set_pipeline(&self, pipeline: RasterPipeline) {
    //     todo!()
    // }

    pub fn dispatch(&self, group_x: u32, group_y: u32, group_z: u32) {
        todo!()
    }

    pub fn dispatch_indirect(&self, info: DispatchIndirectInfo) {
        todo!()
    }


    fn defer_destruction_helper(&mut self, id: GPUResourceId, index: u8) {
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                internal.deferred_destructions.push((id, index));
            },
            CommandListState::Completed(_) => {
                #[cfg(debug_assertions)]
                panic!("Can't record commands on a completed command list.")
            }
        }
    }

    pub fn destroy_buffer_deferred(&mut self, id: BufferId) {
        self.defer_destruction_helper(GPUResourceId(id.0), DEFERRED_DESTRUCTION_BUFFER_INDEX as u8);
    }

    pub fn destroy_image_deferred(&mut self, id: ImageId) {
        self.defer_destruction_helper(GPUResourceId(id.0), DEFERRED_DESTRUCTION_IMAGE_INDEX as u8);
    }

    pub fn destroy_image_view_deferred(&mut self, id: ImageViewId) {
        self.defer_destruction_helper(GPUResourceId(id.0), DEFERRED_DESTRUCTION_IMAGE_VIEW_INDEX as u8);
    }

    pub fn destroy_sampler_deferred(&mut self, id: SamplerId) {
        self.defer_destruction_helper(GPUResourceId(id.0), DEFERRED_DESTRUCTION_SAMPLER_INDEX as u8);
    }


    pub fn begin_renderpass(&self, info: RenderPassBeginInfo) {
        todo!()
    }

    pub fn end_renderpass(&self) {
        todo!()
    }

    // pub fn set_viewport(&self, info: ViewportInfo) {
    //     todo!()
    // }

    pub fn set_scissor(&self, info: vk::Rect2D) {
        todo!()
    }

    pub fn set_depth_bias(&self, info: DepthBiasInfo) {
        todo!()
    }

    pub fn set_index_buffer(&self, id: BufferId, offset: usize, index_type_byte_size: usize) {
        todo!()
    }


    pub fn draw(&self, info: DrawInfo) {
        todo!()
    }

    pub fn draw_indexed(&self, info: DrawIndexedInfo) {
        todo!()
    }

    pub fn draw_indirect(&self, info: DrawIndirectInfo) {
        todo!()
    }

    pub fn draw_indirect_count(&self, info: DrawIndirectCountInfo) {
        todo!()
    }


    pub fn write_timestamp(&mut self, info: WriteTimestampInfo) {
        debug_assert!(info.query_index < info.query_pool.info().query_count, "Write index is out of bounds for the query pool.");
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                internal.flush_barriers();
                unsafe { internal.device.0.logical_device.cmd_write_timestamp(
                    internal.command_buffer,
                    info.stage,
                    info.query_pool.0.timeline_query_pool,
                    info.query_index)
                };
            }
            CommandListState::Completed(_) => panic!("CommandList is already completed.")
        }
    }

    pub fn reset_timestamps(&mut self, info: ResetTimestampsInfo) {
        debug_assert!(info.start_index < info.query_pool.info().query_count, "Reset index is out of bounds for the query pool.");
        match &mut self.0 {
            CommandListState::Recording(internal) => {
                internal.flush_barriers();
                unsafe { internal.device.0.logical_device.cmd_reset_query_pool(
                    internal.command_buffer,
                    info.query_pool.0.timeline_query_pool,
                    info.start_index,
                    info.count)
                };
            }
            CommandListState::Completed(_) => panic!("CommandList is already completed.")
        }
    }


    pub fn complete(self) -> Result<CommandList> {
        match self.0 {
            CommandListState::Recording(mut internal) => {
                internal.flush_barriers();

                unsafe { internal.device.0.logical_device.end_command_buffer(internal.command_buffer).unwrap_unchecked() };

                Ok(CommandList(CommandListState::Completed(
                    Arc::from(internal)
                )))
            },
            CommandListState::Completed(_) => bail!("CommandList is already completed.")
        }
    }

    pub fn is_complete(&self) -> bool {
        match &self.0 {
            CommandListState::Recording(_) => false,
            CommandListState::Completed(_) => true,
        }
    }


    pub fn info(&self) -> &CommandListInfo {
        match &self.0 {
            CommandListState::Recording(internal) => &internal.info,
            CommandListState::Completed(internal) => &internal.info,
        }
    }
}

// CommandList internal methods
impl CommandListInternal {
    fn flush_barriers(&mut self) {
        if self.memory_barrier_batch_count == 0 && self.image_barrier_batch_count == 0 {
            return;
        }

        let dependency_info = vk::DependencyInfo::builder()
            .dependency_flags(vk::DependencyFlags::empty())
            .memory_barriers(&self.memory_barrier_batch[0..self.memory_barrier_batch_count])
            .image_memory_barriers(&self.image_barrier_batch[0..self.image_barrier_batch_count]);

        unsafe {
            self.device.0.logical_device.cmd_pipeline_barrier2(self.command_buffer, &dependency_info);    
        }

        self.memory_barrier_batch_count = 0;
        self.image_barrier_batch_count = 0;
    }
}

impl Drop for CommandListInternal {
    fn drop(&mut self) {
        unsafe {
            self.device.0.logical_device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty()).unwrap_unchecked();

            let main_queue_zombies = &mut self.device.0.main_queue_zombies.lock().unwrap();

            let cpu_timeline = self.device.main_queue_cpu_timeline();

            main_queue_zombies.command_lists.push_front((
                cpu_timeline,
                CommandListZombie {
                    command_buffer: self.command_buffer,
                    command_pool: self.command_pool
                }
            ))
        }
    }
}