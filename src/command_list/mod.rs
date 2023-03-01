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
    pub fn new(
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


    pub fn clear_buffer(&self, info: BufferClearInfo) {
        todo!()
    }

    pub fn clear_image(&self, info: ImageClearInfo) {
        todo!()
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



#[cfg(test)]
mod tests {
    use crate::{core::*, types::*, context::*, device::*, gpu_resources::*, timeline_query::*, split_barrier::*};
    use super::{CommandList, CommandListInfo, info::*};
    use ash::vk;
    use gpu_allocator::MemoryLocation;
    use std::{slice, mem::size_of, borrow::BorrowMut, ops::IndexMut};

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
        let command_list = command_list.complete().unwrap();

        app.device.submit_commands(CommandSubmitInfo {
            command_lists: vec![command_list],
            ..Default::default()
        })
    }

    #[test]
    fn deferred_destruction() {
        let app = App::new();

        let mut command_list = app.device.create_command_list(CommandListInfo {
            debug_name: "deferred_destruction command list"
        }).unwrap();

        let buffer = app.device.create_buffer(BufferInfo {
            size: 4,
            ..Default::default()
        }).unwrap();

        let image = app.device.create_image(ImageInfo {
            size: vk::Extent3D { width: 1, height: 1, depth: 1 },
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        }).unwrap();

        let image_view = app.device.create_image_view(ImageViewInfo {
            image,
            ..Default::default()
        }).unwrap();

        let sampler = app.device.create_sampler(SamplerInfo {
            ..Default::default()
        }).unwrap();

        // The gpu resources are not destroyed here. Their destruction is deferred until the command list completes execution on the gpu.
        command_list.destroy_buffer_deferred(buffer);
        command_list.destroy_image_deferred(image);
        command_list.destroy_image_view_deferred(image_view);
        command_list.destroy_sampler_deferred(sampler);

        // The gpu resources are still alive, as long as this command list is not submitted and has not finished execution.
        let command_list = command_list.complete().unwrap();

        // Even after this call the resources will still be alive, as zombie resources are not checked to be dead in submit calls.
        app.device.submit_commands(CommandSubmitInfo {
            command_lists: vec![command_list],
            ..Default::default()
        });

        app.device.wait_idle();

        // Here the gpu resources will be destroyed.
        // Collect_garbage loops over all zombie resources and destroys them when they are no longer used on the gpu/ their associated command list finished executing.
        app.device.collect_garbage();
    }

    #[test]
    fn copy() {
        let app = App::new();

        let mut command_list = app.device.create_command_list(CommandListInfo {
            debug_name: "copy command list"
        }).unwrap();

        const SIZE_X: u32 = 3;
        const SIZE_Y: u32 = 3;
        const SIZE_Z: u32 = 3;

        type ImageArray = [[[[f32; 4]; SIZE_X as usize]; SIZE_Y as usize]; SIZE_Z as usize];

        let mut data = ImageArray::default();

        for zi in 0..SIZE_Z {
            for yi in 0..SIZE_Y {
                for xi in 0..SIZE_X {
                    data[zi as usize][yi as usize][xi as usize] = [
                        (xi as f32) / ((SIZE_X - 1) as f32),
                        (yi as f32) / ((SIZE_Y - 1) as f32),
                        (zi as f32) / ((SIZE_Z - 1) as f32),
                        1.0
                    ]
                }
            }
        }

        let staging_upload_buffer = app.device.create_buffer(BufferInfo {
            memory_location: MemoryLocation::CpuToGpu,
            size: size_of::<ImageArray>() as u32,
            debug_name: "staging_upload_buffer",
        }).unwrap();

        let device_local_buffer = app.device.create_buffer(BufferInfo {
            memory_location: MemoryLocation::GpuOnly,
            size: size_of::<ImageArray>() as u32,
            debug_name: "device_local_buffer",
        }).unwrap();

        let staging_readback_buffer = app.device.create_buffer(BufferInfo {
            memory_location: MemoryLocation::GpuToCpu,
            size: size_of::<ImageArray>() as u32,
            debug_name: "staging_readback_buffer",
        }).unwrap();

        let image_1 = app.device.create_image(ImageInfo {
            dimensions: match SIZE_Z > 1 { true => 3, false => 2 },
            format: vk::Format::R32G32B32A32_SFLOAT,
            size: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            debug_name: "image_1",
            ..Default::default()
        }).unwrap();

        let image_2 = app.device.create_image(ImageInfo {
            dimensions: match SIZE_Z > 1 { true => 3, false => 2 },
            format: vk::Format::R32G32B32A32_SFLOAT,
            size: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            debug_name: "image_2",
            ..Default::default()
        }).unwrap();

        let timeline_query_pool = app.device.create_timeline_query_pool(TimelineQueryPoolInfo {
            query_count: 2,
            debug_name: "timeline_query"
        }).unwrap();

        let buffer_ptr = unsafe {
            app.device.get_host_address_as::<ImageArray>(staging_upload_buffer)
                .unwrap()
                .as_mut()
        };

        *buffer_ptr = data;

        command_list.reset_timestamps(ResetTimestampsInfo {
            query_pool: timeline_query_pool.clone(),
            start_index: 0,
            count: timeline_query_pool.info().query_count
        });

        command_list.write_timestamp(WriteTimestampInfo {
            query_pool: timeline_query_pool.clone(),
            stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            query_index: 0
        });

        command_list.pipeline_barrier(MemoryBarrierInfo {
            awaited_pipeline_access: access_consts::HOST_WRITE,
            waiting_pipeline_access: access_consts::TRANSFER_READ
        });

        command_list.copy_buffer_to_buffer(BufferCopyInfo {
            src_buffer: staging_upload_buffer,
            dst_buffer: device_local_buffer,
            size: size_of::<ImageArray>() as vk::DeviceSize,
            ..Default::default()
        });

        // Barrier to make sure device_local_buffer is has no read after write hazard.
        command_list.pipeline_barrier(MemoryBarrierInfo {
            awaited_pipeline_access: access_consts::TRANSFER_WRITE,
            waiting_pipeline_access: access_consts::TRANSFER_READ
        });

        command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
            awaited_pipeline_access: access_consts::TRANSFER_WRITE,
            waiting_pipeline_access: access_consts::TRANSFER_WRITE,
            after_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: image_1,
            ..Default::default()
        });

        command_list.copy_buffer_to_image(BufferImageCopyInfo {
            buffer: device_local_buffer,
            image: image_1,
            image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image_extent: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
            ..Default::default()
        });

        command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
            awaited_pipeline_access: access_consts::TRANSFER_WRITE,
            waiting_pipeline_access: access_consts::TRANSFER_READ,
            after_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image: image_1,
            ..Default::default()
        });

        command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
            waiting_pipeline_access: access_consts::TRANSFER_WRITE,
            after_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: image_2,
            ..Default::default()
        });

        command_list.copy_image_to_image(ImageCopyInfo {
            src_image: image_1,
            src_image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_image: image_2,
            dst_image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            extent: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
            ..Default::default()
        });

        command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
            awaited_pipeline_access: access_consts::TRANSFER_WRITE,
            waiting_pipeline_access: access_consts::TRANSFER_READ,
            after_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image: image_2,
            ..Default::default()
        });

        // Barrier to make sure device_local_buffer is has no write after read hazard.
        command_list.pipeline_barrier(MemoryBarrierInfo {
            awaited_pipeline_access: access_consts::TRANSFER_READ,
            waiting_pipeline_access: access_consts::TRANSFER_WRITE
        });

        command_list.copy_image_to_buffer(ImageBufferCopyInfo {
            image: image_2,
            image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image_extent: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
            buffer: device_local_buffer,
            ..Default::default()
        });

        // Barrier to make sure device_local_buffer is has no read after write hazard.
        command_list.pipeline_barrier(MemoryBarrierInfo {
            awaited_pipeline_access: access_consts::TRANSFER_WRITE,
            waiting_pipeline_access: access_consts::TRANSFER_READ
        });

        command_list.copy_buffer_to_buffer(BufferCopyInfo {
            src_buffer: device_local_buffer,
            dst_buffer: staging_readback_buffer,
            size: size_of::<ImageArray>() as vk::DeviceSize,
            ..Default::default()
        });

        // Barrier to make sure staging_readback_buffer is has no read after write hazard.
        command_list.pipeline_barrier(MemoryBarrierInfo {
            awaited_pipeline_access: access_consts::TRANSFER_WRITE,
            waiting_pipeline_access: access_consts::HOST_READ
        });

        command_list.write_timestamp(WriteTimestampInfo {
            query_pool: timeline_query_pool.clone(),
            stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            query_index: 1
        });

        let command_list = command_list.complete().unwrap();

        app.device.submit_commands(CommandSubmitInfo {
            command_lists: vec![command_list],
            ..Default::default()
        });

        app.device.wait_idle();

        // Validate and display results

        let query_results = timeline_query_pool.get_query_results(0, 2).unwrap();
        if query_results[0].1 != 0 && query_results[1].1 != 0 {
            println!("GPU execution took {} ms", ((query_results[1].0 - query_results[0].0) as f64) / 1000000.0);
        }

        let readback_data = unsafe {
            app.device.get_host_address_as::<ImageArray>(staging_upload_buffer)
                .unwrap()
                .as_ref()
        };

        fn get_printable_char_buffer(in_data: &ImageArray) -> String {
            const PIXEL: &str = "\x1B[48;2;000;000;000m  ";
            const LINE_TERMINATOR: &str = "\x1B[0m ";
            const NEWLINE_TERMINATOR: &str = "\x1B[0m\n";

            let capacity: usize = (SIZE_X * SIZE_Y * SIZE_Z) as usize * (PIXEL.len() - 1)
                                + (SIZE_Y * SIZE_Z) as usize * (LINE_TERMINATOR.len() - 1)
                                + SIZE_Z as usize * (NEWLINE_TERMINATOR.len() - 1)
                                + 1;
            let mut data = String::with_capacity(capacity);

            for zi in 0..SIZE_Z as usize {
                for yi in 0..SIZE_Y as usize {
                    for xi in 0..SIZE_X as usize {
                        let r = (in_data[zi][yi][xi][0] * 255.0) as u8;
                        let g = (in_data[zi][yi][xi][1] * 255.0) as u8;
                        let b = (in_data[zi][yi][xi][2] * 255.0) as u8;
                        let mut next_pixel = String::from(PIXEL).into_bytes();
                        next_pixel[7 + 0 * 4 + 0] = 48 + (r / 100);
                        next_pixel[7 + 0 * 4 + 1] = 48 + ((r % 100) / 10);
                        next_pixel[7 + 0 * 4 + 2] = 48 + (r % 10);
                        next_pixel[7 + 1 * 4 + 0] = 48 + (g / 100);
                        next_pixel[7 + 1 * 4 + 1] = 48 + ((g % 100) / 10);
                        next_pixel[7 + 1 * 4 + 2] = 48 + (g % 10);
                        next_pixel[7 + 2 * 4 + 0] = 48 + (b / 100);
                        next_pixel[7 + 2 * 4 + 1] = 48 + ((b % 100) / 10);
                        next_pixel[7 + 2 * 4 + 2] = 48 + (b % 10);
                        let next_pixel = String::from_utf8(next_pixel).unwrap();
                        data.push_str(&next_pixel);
                    }
                    data.push_str(LINE_TERMINATOR);
                }
                data.push_str(NEWLINE_TERMINATOR);
            }
            
            data.to_ascii_lowercase()
        }

        println!("Original data:\n{}", get_printable_char_buffer(&data));
        println!("Readback data:\n{}", get_printable_char_buffer(&readback_data));

        #[cfg(debug_assertions)]
        for zi in 0..SIZE_Z {
            for yi in 0..SIZE_Y {
                for xi in 0..SIZE_X {
                    for ci in 0..4 {
                        debug_assert_eq!(
                            data[zi as usize][yi as usize][xi as usize][ci as usize],
                            readback_data[zi as usize][yi as usize][xi as usize][ci as usize],
                            "Readback data differs from upload data."
                        )
                    }
                }
            }
        }

        app.device.destroy_buffer(staging_upload_buffer);
        app.device.destroy_buffer(device_local_buffer);
        app.device.destroy_buffer(staging_readback_buffer);
        app.device.destroy_image(image_1);
        app.device.destroy_image(image_2);

        app.device.collect_garbage();
    }

}