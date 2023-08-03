mod info;
pub use info::*;

use crate::{
    core::*,
    device::{Device, DeviceInternal},
    gpu_resources::{
        GPUResourceId,
        BufferId,
        ImageId,
        ImageViewId,
        SamplerId
    },
    split_barrier::*,
    pipeline::*
};
use anyhow::{Result, bail};
use ash::vk::{self, Handle};
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
    pipeline_layouts: [vk::PipelineLayout; PIPELINE_LAYOUT_COUNT as usize],
    pub deferred_destructions: Vec<(GPUResourceId, u8)>,
    constant_buffer_bindings: [ConstantBufferInfo; CONSTANT_BUFFER_BINDINGS_COUNT as usize]
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

        let pipeline_layouts = device.0.gpu_shader_resource_table.pipeline_layouts;

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
            pipeline_layouts,
            deferred_destructions: vec![],
            constant_buffer_bindings: Default::default()
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
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

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
    }

    pub fn copy_buffer_to_image(&mut self, info: BufferImageCopyInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

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
    }

    pub fn copy_image_to_buffer(&mut self, info: ImageBufferCopyInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

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
    }

    pub fn copy_image_to_image(&mut self, info: ImageCopyInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

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
    }

    pub fn blit_image_to_image(&mut self, info: ImageBlitInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

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
    }


    pub fn clear_buffer(&mut self, info: BufferClearInfo) {
        let CommandListState::Recording(command_list) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
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
            #[cfg(not(debug_assertions))]
            unreachable!();
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
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        if internal.memory_barrier_batch_count == COMMAND_LIST_BARRIER_MAX_BATCH_SIZE {
            internal.flush_barriers();
        }

        internal.memory_barrier_batch[internal.memory_barrier_batch_count] = vk::MemoryBarrier2 {
            src_stage_mask: info.src_access.0,
            src_access_mask: info.src_access.1,
            dst_stage_mask: info.dst_access.0,
            dst_access_mask: info.dst_access.1,
            ..Default::default()
        };

        internal.memory_barrier_batch_count += 1;
    }

    pub fn pipeline_barrier_image_transition(&mut self, info: ImageBarrierInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        if internal.image_barrier_batch_count == COMMAND_LIST_BARRIER_MAX_BATCH_SIZE {
            internal.flush_barriers();
        }

        internal.image_barrier_batch[internal.image_barrier_batch_count] = vk::ImageMemoryBarrier2 {
            src_stage_mask: info.src_access.0,
            src_access_mask: info.src_access.1,
            dst_stage_mask: info.dst_access.0,
            dst_access_mask: info.dst_access.1,
            old_layout: info.src_layout,
            new_layout: info.dst_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: internal.device.0.image_slot(info.image).image,
            subresource_range: info.range,
            ..Default::default()
        };

        internal.image_barrier_batch_count += 1;
    }

    pub fn wait_split_barriers(&mut self, infos: &[SplitBarrierWaitInfo]) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        // TODO: thread locals?
        #[derive(Default)]
        struct SplitBarrierDependencyInfoBuffer {
            image_memory_barriers: Vec<vk::ImageMemoryBarrier2>,
            memory_barriers: Vec<vk::MemoryBarrier2>
        }
        let mut split_barrier_dependency_infos_aux_buffer: Vec<SplitBarrierDependencyInfoBuffer> = vec![];
        let mut split_barrier_dependency_infos_buffer = vec![];
        let mut split_barrier_events_buffer = vec![];

        internal.flush_barriers();
        for end_info in infos {
            split_barrier_dependency_infos_aux_buffer.push(SplitBarrierDependencyInfoBuffer::default());
            let dependency_info_aux_buffer = split_barrier_dependency_infos_aux_buffer.last_mut().unwrap();
            for image_barrier in end_info.image_barriers.iter() {
                dependency_info_aux_buffer.image_memory_barriers.push(vk::ImageMemoryBarrier2 {
                    src_stage_mask: image_barrier.src_access.0,
                    src_access_mask: image_barrier.src_access.1,
                    dst_stage_mask: image_barrier.dst_access.0,
                    dst_access_mask: image_barrier.dst_access.1,
                    old_layout: image_barrier.src_layout,
                    new_layout: image_barrier.dst_layout,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: internal.device.0.image_slot(image_barrier.image).image,
                    subresource_range: image_barrier.range,
                    ..Default::default()
                });
            }
            for memory_barrier in end_info.memory_barriers.iter() {
                dependency_info_aux_buffer.memory_barriers.push(vk::MemoryBarrier2 {
                    src_stage_mask: memory_barrier.src_access.0,
                    src_access_mask: memory_barrier.src_access.1,
                    dst_stage_mask: memory_barrier.dst_access.0,
                    dst_access_mask: memory_barrier.dst_access.1,
                    ..Default::default()
                });
            }
            split_barrier_dependency_infos_buffer.push(vk::DependencyInfo {
                memory_barrier_count: dependency_info_aux_buffer.memory_barriers.len() as u32,
                p_memory_barriers: dependency_info_aux_buffer.memory_barriers.as_ptr(),
                image_memory_barrier_count: dependency_info_aux_buffer.image_memory_barriers.len() as u32,
                p_image_memory_barriers: dependency_info_aux_buffer.image_memory_barriers.as_ptr(),
                ..Default::default()
            });
            split_barrier_events_buffer.push(vk::Event::from_raw(end_info.split_barrier.data));
        }
        unsafe {internal.device.0.logical_device.cmd_wait_events2(
            internal.command_buffer, 
            &split_barrier_events_buffer, 
            &split_barrier_dependency_infos_buffer
        )};

        split_barrier_dependency_infos_aux_buffer.clear();
        split_barrier_dependency_infos_buffer.clear();
        split_barrier_events_buffer.clear();
    }

    pub fn wait_split_barrier(&mut self, info: SplitBarrierWaitInfo) {
        self.wait_split_barriers(&[info]);
    }

    pub fn signal_split_barrier(&mut self, info: SplitBarrierSignalInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        // TODO: thread locals?
        #[derive(Default)]
        struct SplitBarrierDependencyInfoBuffer {
            image_memory_barriers: Vec<vk::ImageMemoryBarrier2>,
            memory_barriers: Vec<vk::MemoryBarrier2>
        }
        let mut split_barrier_dependency_info_aux_buffer: Vec<SplitBarrierDependencyInfoBuffer> = vec![];
        // let mut split_barrier_dependency_info_buffer = vec![];
        // let mut split_barrier_events_buffer = vec![];

        internal.flush_barriers();

        split_barrier_dependency_info_aux_buffer.push(Default::default());
        let dependency_info_aux_buffer = split_barrier_dependency_info_aux_buffer.last_mut().unwrap();

        for image_barrier in info.image_barriers.iter() {
            dependency_info_aux_buffer.image_memory_barriers.push(vk::ImageMemoryBarrier2 {
                src_stage_mask: image_barrier.src_access.0,
                src_access_mask: image_barrier.src_access.1,
                dst_stage_mask: image_barrier.dst_access.0,
                dst_access_mask: image_barrier.dst_access.1,
                old_layout: image_barrier.src_layout,
                new_layout: image_barrier.dst_layout,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: internal.device.0.image_slot(image_barrier.image).image,
                subresource_range: image_barrier.range,
                ..Default::default()
            })
        }
        for memory_barrier in info.memory_barriers.iter() {
            dependency_info_aux_buffer.memory_barriers.push(vk::MemoryBarrier2 {
                src_stage_mask: memory_barrier.src_access.0,
                src_access_mask: memory_barrier.src_access.1,
                dst_stage_mask: memory_barrier.dst_access.0,
                dst_access_mask: memory_barrier.dst_access.1,
                ..Default::default()
            });
        }

        let dependency_info = vk::DependencyInfo {
            memory_barrier_count: dependency_info_aux_buffer.memory_barriers.len() as u32,
            p_memory_barriers: dependency_info_aux_buffer.memory_barriers.as_ptr(),
            image_memory_barrier_count: dependency_info_aux_buffer.image_memory_barriers.len() as u32,
            p_image_memory_barriers: dependency_info_aux_buffer.image_memory_barriers.as_ptr(),
            ..Default::default()
        };
        unsafe { internal.device.0.logical_device.cmd_set_event2(
            internal.command_buffer,
            vk::Event::from_raw(info.split_barrier.data),
            &dependency_info
        ) };
        split_barrier_dependency_info_aux_buffer.clear();
    }

    pub fn reset_split_barrier(&mut self, info: ResetSplitBarrierInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };
        internal.flush_barriers();
        unsafe { internal.device.0.logical_device.cmd_reset_event2(
            internal.command_buffer,
            vk::Event::from_raw(info.barrier.data),
            info.stage
        ) };
    }


    // pub fn push_constant(&self, data: *const c_void, size: u32, offset: u32) {
    //     todo!()
    // }

    pub fn set_push_constant<T>(&mut self, data: &T, offset: u32) {
        let size = std::mem::size_of::<T>();
        debug_assert!(size <= MAX_PUSH_CONSTANT_BYTE_SIZE as usize, "{}", MAX_PUSH_CONSTANT_SIZE_ERROR);
        debug_assert!(size % 4 == 0, "Push constant must have an alignment of 4.");

        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();

        unsafe {
            let (_, bytes, _) = slice::from_ref(data).align_to::<u8>();

            internal.device.0.logical_device.cmd_push_constants(
                internal.command_buffer,
                internal.pipeline_layouts[(size + 3) / 4],
                vk::ShaderStageFlags::ALL,
                offset,
                bytes
            )
        }
    }

    pub fn set_constant_buffer(&mut self, info: ConstantBufferInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        debug_assert!(info.size > 0, "Constant buffer size must be greater than 0.");
        debug_assert!(
            info.offset as u64 % internal.device.0.properties.limits.min_uniform_buffer_offset_alignment == 0,
            "Constant buffer offset must respect uniform buffer alignment requirements."
        );

        let buffer_size = internal.device.0.buffer_slot(info.buffer).info.size;

        debug_assert!(
            info.size + info.offset <= buffer_size as u64,
            "Constant buffer size, offset ({}, {}) must describe a valid range within the given buffer with size {}.",
            info.size, info.offset, buffer_size
        );
        let slot = info.slot;
        debug_assert!(
            slot < CONSTANT_BUFFER_BINDINGS_COUNT as u32,
            "Constant buffer slot must be in the range 0-{}.",
            CONSTANT_BUFFER_BINDINGS_COUNT
        );

        internal.constant_buffer_bindings[slot as usize] = info;
    }

    pub fn set_compute_pipeline(&mut self, pipeline: ComputePipeline) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();
        internal.flush_constant_buffer_bindings(vk::PipelineBindPoint::COMPUTE, pipeline.0.pipeline_layout);

        unsafe {
            let logical_device = &internal.device.0.logical_device;

            logical_device.cmd_bind_descriptor_sets(
                internal.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.0.pipeline_layout,
                0,
                slice::from_ref(&internal.device.0.gpu_shader_resource_table.descriptor_set),
                &[]
            );

            logical_device.cmd_bind_pipeline(internal.command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline.0.pipeline);
        };
    }

    pub fn set_raster_pipeline(&mut self, pipeline: &RasterPipeline) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();
        internal.flush_constant_buffer_bindings(vk::PipelineBindPoint::GRAPHICS, pipeline.0.pipeline_layout);

        unsafe {
            let logical_device = &internal.device.0.logical_device;

            logical_device.cmd_bind_descriptor_sets(
                internal.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.0.pipeline_layout,
                0,
                slice::from_ref(&internal.device.0.gpu_shader_resource_table.descriptor_set),
                &[]
            );

            logical_device.cmd_bind_pipeline(internal.command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.0.pipeline);
        };
    }

    pub fn dispatch(&mut self, group_x: u32, group_y: u32, group_z: u32) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();

        unsafe {
            internal.device.0.logical_device.cmd_dispatch(internal.command_buffer, group_x, group_y, group_z);
        };
    }

    pub fn dispatch_indirect(&mut self, info: DispatchIndirectInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();

        unsafe {
            internal.device.0.logical_device.cmd_dispatch_indirect(internal.command_buffer, internal.device.0.buffer_slot(info.indirect_buffer).buffer, info.offset);
        };
    }


    fn defer_destruction_helper(&mut self, id: GPUResourceId, index: u8) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.deferred_destructions.push((id, index));
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


    pub fn begin_renderpass(&mut self, info: RenderPassBeginInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();

        let fill_rendering_attachment_info = |info: &RenderAttachmentInfo| -> vk::RenderingAttachmentInfo {
            debug_assert!(!info.image_view.is_empty(), "Must provide valid image view to render attachment.");

            vk::RenderingAttachmentInfo::builder()
                .image_view(internal.device.0.image_view_slot(info.image_view).image_view)
                .image_layout(info.layout)
                .resolve_mode(vk::ResolveModeFlags::NONE)
                .resolve_image_view(vk::ImageView::null())
                .resolve_image_layout(vk::ImageLayout::UNDEFINED)
                .load_op(info.load_op)
                .store_op(info.store_op)
                .clear_value(info.clear_value)
                .build()
        };

        debug_assert!(
            info.color_attachments.len() <= COMMAND_LIST_COLOR_ATTACHMENT_MAX,
            "Too many color attachments. Make a pull request to bump max."
        );
        let color_attachment_infos = info.color_attachments.iter().map(|color_info| {
            fill_rendering_attachment_info(color_info)
        })
        .collect::<Vec<vk::RenderingAttachmentInfo>>();

        let depth_attachment_info = match &info.depth_attachment {
            Some(depth_info) => fill_rendering_attachment_info(&depth_info),
            None => vk::RenderingAttachmentInfo::default()
        };

        let stencil_attachment_info = match &info.stencil_attachment {
            Some(stencil_info) => fill_rendering_attachment_info(&stencil_info),
            None => vk::RenderingAttachmentInfo::default()
        };

        let mut rendering_info = vk::RenderingInfo::builder()
            .render_area(info.render_area)
            .layer_count(1)
            .color_attachments(&color_attachment_infos);
        if info.depth_attachment.is_some() {
            rendering_info = rendering_info.depth_attachment(&depth_attachment_info);
        }
        if info.stencil_attachment.is_some() {
            rendering_info = rendering_info.stencil_attachment(&stencil_attachment_info);
        }

        let logical_device = &internal.device.0.logical_device;
        unsafe { logical_device.cmd_set_scissor(internal.command_buffer, 0, slice::from_ref(&info.render_area)) };

        let viewport = vk::Viewport::builder()
            .x(info.render_area.offset.x as f32)
            .y(info.render_area.offset.y as f32)
            .width(info.render_area.extent.width as f32)
            .height(info.render_area.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();

        unsafe { logical_device.cmd_set_viewport(internal.command_buffer, 0, slice::from_ref(&viewport)) };
        
        unsafe { logical_device.cmd_begin_rendering(internal.command_buffer, &rendering_info) };
    }

    pub fn end_renderpass(&mut self) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();

        unsafe { internal.device.0.logical_device.cmd_end_rendering(internal.command_buffer) };
    }

    pub fn set_viewport(&mut self, info: vk::Viewport) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();

        unsafe { internal.device.0.logical_device.cmd_set_viewport(internal.command_buffer, 0, slice::from_ref(&info)) };
    }

    pub fn set_scissor(&mut self, info: vk::Rect2D) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();

        unsafe { internal.device.0.logical_device.cmd_set_scissor(internal.command_buffer, 0, slice::from_ref(&info)) };
    }

    pub fn set_depth_bias(&mut self, info: DepthBiasInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();

        unsafe { internal.device.0.logical_device.cmd_set_depth_bias(internal.command_buffer, info.constant_factor, info.clamp, info.slope_factor) };
    }

    pub fn set_index_buffer(&self, id: BufferId, offset: usize, index_type_byte_size: usize) {
        todo!()
    }


    pub fn draw(&self, info: DrawInfo) {
        let CommandListState::Recording(internal) = &self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        unsafe {
            internal.device.0.logical_device.cmd_draw(
                internal.command_buffer,
                info.vertex_count,
                info.instance_count,
                info.first_vertex,
                info.first_instance
            )
        }
    }

    pub fn draw_indexed(&self, info: DrawIndexedInfo) {
        let CommandListState::Recording(internal) = &self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        unsafe {
            internal.device.0.logical_device.cmd_draw_indexed(
                internal.command_buffer,
                info.index_count,
                info.instance_count,
                info.first_index,
                info.vertex_offset,
                info.first_instance
            )
        }
    }

    pub fn draw_indirect(&self, info: DrawIndirectInfo) {
        let CommandListState::Recording(internal) = &self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        unsafe {
            if info.is_indexed {
                internal.device.0.logical_device.cmd_draw_indexed_indirect(
                    internal.command_buffer,
                    internal.device.0.buffer_slot(info.draw_command_buffer).buffer,
                    info.draw_command_buffer_read_offset,
                    info.draw_count,
                    info.draw_command_stride
                );
            } else {
                internal.device.0.logical_device.cmd_draw_indirect(
                    internal.command_buffer,
                    internal.device.0.buffer_slot(info.draw_command_buffer).buffer,
                    info.draw_command_buffer_read_offset,
                    info.draw_count,
                    info.draw_command_stride
                );
            }
        }
    }

    pub fn draw_indirect_count(&self, info: DrawIndirectCountInfo) {
        let CommandListState::Recording(internal) = &self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        unsafe {
            if info.is_indexed {
                internal.device.0.logical_device.cmd_draw_indexed_indirect_count(
                    internal.command_buffer,
                    internal.device.0.buffer_slot(info.draw_command_buffer).buffer,
                    info.draw_command_buffer_read_offset,
                    internal.device.0.buffer_slot(info.draw_count_buffer).buffer,
                    info.draw_count_buffer_read_offset,
                    info.max_draw_count,
                    info.draw_command_stride
                );
            } else {
                internal.device.0.logical_device.cmd_draw_indirect_count(
                    internal.command_buffer,
                    internal.device.0.buffer_slot(info.draw_command_buffer).buffer,
                    info.draw_command_buffer_read_offset,
                    internal.device.0.buffer_slot(info.draw_count_buffer).buffer,
                    info.draw_count_buffer_read_offset,
                    info.max_draw_count,
                    info.draw_command_stride
                );
            }
        }
    }


    pub fn write_timestamp(&mut self, info: WriteTimestampInfo) {
        debug_assert!(info.query_index < info.query_pool.info().query_count, "Write index is out of bounds for the query pool.");

        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();
        unsafe { internal.device.0.logical_device.cmd_write_timestamp(
            internal.command_buffer,
            info.stage,
            info.query_pool.0.timeline_query_pool,
            info.query_index)
        };
    }

    pub fn reset_timestamps(&mut self, info: ResetTimestampsInfo) {
        debug_assert!(info.start_index < info.query_pool.info().query_count, "Reset index is out of bounds for the query pool.");

        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();
        unsafe { internal.device.0.logical_device.cmd_reset_query_pool(
            internal.command_buffer,
            info.query_pool.0.timeline_query_pool,
            info.start_index,
            info.count)
        };
    }

    pub fn begin_label(&mut self, info: CommandLabelInfo) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();
        let debug_label_info = vk::DebugUtilsLabelEXT {
            p_label_name: info.label_name.as_ptr() as *const i8,
            color: info.label_color,
            ..Default::default()
        };
        unsafe { internal.device.0.context.debug_utils().cmd_begin_debug_utils_label(internal.command_buffer, &debug_label_info) };
    }

    pub fn end_label(&mut self) {
        let CommandListState::Recording(internal) = &mut self.0 else {
            #[cfg(debug_assertions)]
            panic!("Can't record commands on a completed command list.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();
        unsafe { internal.device.0.context.debug_utils().cmd_end_debug_utils_label(internal.command_buffer); }
    }


    pub fn complete(self) -> CommandList {
        let CommandListState::Recording(mut internal) = self.0 else {
            #[cfg(debug_assertions)]
            panic!("CommandList is already completed.");
            #[cfg(not(debug_assertions))]
            unreachable!();
        };

        internal.flush_barriers();

        unsafe { internal.device.0.logical_device.end_command_buffer(internal.command_buffer).unwrap_unchecked() };

        CommandList(CommandListState::Completed(
            Arc::from(internal)
        ))
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

    fn flush_constant_buffer_bindings(&mut self, bind_point: vk::PipelineBindPoint, pipeline_layout: vk::PipelineLayout) {
        let mut descriptor_buffer_info: [vk::DescriptorBufferInfo; CONSTANT_BUFFER_BINDINGS_COUNT as usize] = Default::default();
        let mut descriptor_writes: [vk::WriteDescriptorSet; CONSTANT_BUFFER_BINDINGS_COUNT as usize] = Default::default();

        for (index, constant_buffer_info) in self.constant_buffer_bindings.iter().enumerate() {
            if constant_buffer_info.buffer.is_empty() {
                descriptor_buffer_info[index] = vk::DescriptorBufferInfo::default()
            } else {
                descriptor_buffer_info[index] = vk::DescriptorBufferInfo {
                    buffer: self.device.0.buffer_slot(constant_buffer_info.buffer).buffer,
                    offset: constant_buffer_info.offset,
                    range: constant_buffer_info.size
                }
            }

            descriptor_writes[index] = vk::WriteDescriptorSet {
                dst_binding: index as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_buffer_info: &descriptor_buffer_info[index],
                ..Default::default()
            }
        }

        unsafe { self.device.0.push_descriptor.cmd_push_descriptor_set(
            self.command_buffer, 
            bind_point, 
            pipeline_layout, 
            CONSTANT_BUFFER_BINDINGS_SET, 
            &descriptor_writes
        )}

        self.constant_buffer_bindings.fill(Default::default());
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