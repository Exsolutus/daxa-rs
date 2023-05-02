use crate::{timeline_query::*, split_barrier::*};
use super::{
    BufferId,
    ImageId,
    ImageViewId,
};

use ash::vk;

use std::{
    borrow::Cow,
};

// reexport
pub use {
    vk::{
        AttachmentLoadOp,
        ImageLayout,
        ClearValue,
        ClearColorValue,
        ClearDepthStencilValue
    },
};


#[derive(Default)]
pub struct CommandListInfo {
    pub debug_name: Cow<'static, str>,
}


#[derive(Clone, Copy, Default)]
pub struct ConstantBufferInfo {
    pub slot: u32, // Binding slot the buffer will be bound to.
    pub buffer: BufferId,
    pub size: vk::DeviceSize,
    pub offset: vk::DeviceSize
}


pub struct ImageBlitInfo {
    pub src_image: ImageId,
    pub src_image_layout: vk::ImageLayout,
    pub dst_image: ImageId,
    pub dst_image_layout: vk::ImageLayout,
    pub src_layers: vk::ImageSubresourceLayers,
    pub src_offsets: [vk::Offset3D; 2],
    pub dst_layers: vk::ImageSubresourceLayers,
    pub dst_offsets: [vk::Offset3D; 2],
    pub filter: vk::Filter
}

impl Default for ImageBlitInfo {
    fn default() -> Self {
        Self {
            src_image: Default::default(),
            src_image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_image: Default::default(),
            dst_image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_layers: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            },
            src_offsets: Default::default(),
            dst_layers: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            },
            dst_offsets: Default::default(),
            filter: Default::default()
        }
    }
}


#[derive(Default)]
pub struct BufferCopyInfo {
    pub src_buffer: BufferId,
    pub src_offset: vk::DeviceSize,
    pub dst_buffer: BufferId,
    pub dst_offset: vk::DeviceSize,
    pub size: vk::DeviceSize
}


pub struct BufferImageCopyInfo {
    pub buffer: BufferId,
    pub buffer_offset: vk::DeviceSize,
    pub image: ImageId,
    pub image_layout: vk::ImageLayout,
    pub image_layers: vk::ImageSubresourceLayers,
    pub image_offset: vk::Offset3D,
    pub image_extent: vk::Extent3D
}

impl Default for BufferImageCopyInfo {
    fn default() -> Self {
        Self {
            buffer: Default::default(),
            buffer_offset: Default::default(),
            image: Default::default(),
            image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image_layers: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            },
            image_offset: Default::default(),
            image_extent: Default::default()
        }
    }
}


pub struct ImageBufferCopyInfo {
    pub image: ImageId,
    pub image_layout: vk::ImageLayout,
    pub image_layers: vk::ImageSubresourceLayers,
    pub image_offset: vk::Offset3D,
    pub image_extent: vk::Extent3D,
    pub buffer: BufferId,
    pub buffer_offset: vk::DeviceSize
}

impl Default for ImageBufferCopyInfo {
    fn default() -> Self {
        Self {
            image: Default::default(),
            image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image_layers: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            },
            image_offset: Default::default(),
            image_extent: Default::default(),
            buffer: Default::default(),
            buffer_offset: Default::default()
        }
    }
}


pub struct ImageCopyInfo {
    pub src_image: ImageId,
    pub src_image_layout: vk::ImageLayout,
    pub src_layers: vk::ImageSubresourceLayers,
    pub src_offset: vk::Offset3D,
    pub dst_image: ImageId,
    pub dst_image_layout: vk::ImageLayout,
    pub dst_layers: vk::ImageSubresourceLayers,
    pub dst_offset: vk::Offset3D,
    pub extent: vk::Extent3D,
}

impl Default for ImageCopyInfo {
    fn default() -> Self {
        Self {
            src_image: Default::default(),
            src_image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_layers: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            },
            src_offset: Default::default(),
            dst_image: Default::default(),
            dst_image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            dst_layers: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            },
            dst_offset: Default::default(),
            extent: Default::default(),
        }
    }
}


pub struct ImageClearInfo {
    pub image: ImageId,
    pub image_layout: vk::ImageLayout,
    pub image_range: vk::ImageSubresourceRange,
    pub clear_value: vk::ClearValue
}

impl Default for ImageClearInfo {
    fn default() -> Self {
        Self {
            image: Default::default(),
            image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            },
            clear_value: Default::default()
        }
    }
}


#[derive(Default)]
pub struct BufferClearInfo {
    pub buffer: BufferId,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    pub clear_value: u32
}


pub struct RenderAttachmentInfo {
    pub image_view: ImageViewId,
    pub layout: vk::ImageLayout,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: vk::ClearValue
}

impl Default for RenderAttachmentInfo {
    fn default() -> Self {
        Self {
            image_view: Default::default(),
            layout: vk::ImageLayout::ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: Default::default()
        }
    }
}


#[derive(Default)]
pub struct RenderPassBeginInfo {
    pub color_attachments: Vec<RenderAttachmentInfo>,
    pub depth_attachment: Option<RenderAttachmentInfo>,
    pub stencil_attachment: Option<RenderAttachmentInfo>,
    pub render_area: vk::Rect2D
}


#[derive(Default)]
pub struct DispatchIndirectInfo {
    pub indirect_buffer: BufferId,
    pub offset: vk::DeviceSize
}


pub struct DrawInfo {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32
}

impl Default for DrawInfo {
    fn default() -> Self {
        Self {
            vertex_count: 0,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0
        }
    }
}


pub struct DrawIndexedInfo {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32
}

impl Default for DrawIndexedInfo {
    fn default() -> Self {
        Self {
            index_count: 0,
            instance_count: 1,
            first_index: 0,
            vertex_offset: 0,
            first_instance: 0
        }
    }
}


pub struct DrawIndirectInfo {
    pub draw_command_buffer: BufferId,
    pub draw_command_buffer_read_offset: vk::DeviceSize,
    pub draw_count: u32,
    pub draw_command_stride: u32,
    pub is_indexed: bool
}

impl Default for DrawIndirectInfo {
    fn default() -> Self {
        Self {
            draw_command_buffer: Default::default(),
            draw_command_buffer_read_offset: 0,
            draw_count: 1,
            draw_command_stride: 0,
            is_indexed: false
        }
    }
}


pub struct DrawIndirectCountInfo {
    pub draw_command_buffer: BufferId,
    pub draw_command_buffer_read_offset: vk::DeviceSize,
    pub draw_count_buffer: BufferId,
    pub draw_count_buffer_read_offset: vk::DeviceSize,
    pub max_draw_count: u32,
    pub draw_command_stride: u32,
    pub is_indexed: bool
}

impl Default for DrawIndirectCountInfo {
    fn default() -> Self {
        Self {
            draw_command_buffer: Default::default(),
            draw_command_buffer_read_offset: 0,
            draw_count_buffer: Default::default(),
            draw_count_buffer_read_offset: 0,
            max_draw_count: u16::MAX as u32,
            draw_command_stride: 0,
            is_indexed: false
        }
    }
}


pub struct ResetSplitBarriersInfo {
    pub split_barrier: SplitBarrierState,
    pub stage: vk::PipelineStageFlags
}


pub struct WaitSplitBarriersInfo {
    pub split_barriers: [SplitBarrierState]
}


pub struct WriteTimestampInfo {
    pub query_pool: TimelineQueryPool,
    pub stage: vk::PipelineStageFlags,
    pub query_index: u32
}


pub struct ResetTimestampsInfo {
    pub query_pool: TimelineQueryPool,
    pub start_index: u32,
    pub count: u32
}


pub struct ResetSplitBarrierInfo {
    pub barrier: SplitBarrierState,
    pub stage: vk::PipelineStageFlags
}


#[derive(Default)]
pub struct DepthBiasInfo {
    pub constant_factor: f32,
    pub clamp: f32,
    pub slope_factor: f32
}
