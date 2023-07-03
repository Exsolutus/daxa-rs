mod resource_table;
mod resource_pool;

pub(crate) use resource_table::*;
pub(crate) use resource_pool::*;

use crate::{
    core::*,
    memory_block::AllocationInfo
};

use ash::vk;
use bitfield::*;
use std::{
    borrow::Cow,
    fmt::Display,
};

// reexport
pub use {
    gpu_allocator::MemoryLocation,
    vk::ImageSubresourceRange,
    vk::ImageAspectFlags
};



pub const BUFFER_BINDING: u32 = 0;
pub const STORAGE_IMAGE_BINDING: u32 = 1;
pub const SAMPLED_IMAGE_BINDING: u32 = 2;
pub const SAMPLER_BINDING: u32 = 3;
pub const BUFFER_DEVICE_ADDRESS_BUFFER_BINDING: u32 = 4;

pub const NOT_OWNED_BY_SWAPCHAIN: i32 = -1;



#[derive(ResourceId, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct GPUResourceId(pub(crate) u32);

#[derive(ResourceId, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct BufferId(pub u32);

impl Display for BufferId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "index: {}\nversion: {}", self.index(), self.version())
    }
}

#[derive(ResourceId, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct ImageId(pub(crate) u32);

impl ImageId {
    pub fn default_view(&self) -> ImageViewId {
        ImageViewId(self.0)
    }
}

impl Display for ImageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "index: {}\nversion: {}", self.index(), self.version())
    }
}

#[derive(ResourceId, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct ImageViewId(pub(crate) u32);

#[derive(ResourceId, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct SamplerId(pub(crate) u32);



#[derive(Clone, Debug)]
pub struct BufferInfo {
    pub size: u32,
    pub allocation_info: AllocationInfo,
    pub debug_name: Cow<'static, str>,
}

impl Default for BufferInfo {
    fn default() -> Self {
        Self {
            size: 0,
            allocation_info: AllocationInfo::Automatic(MemoryLocation::GpuOnly),
            debug_name: "".into()
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImageInfo {
    pub dimensions: u32,
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
    pub size: vk::Extent3D,
    pub mip_level_count: u32,
    pub array_layer_count: u32,
    pub sample_count: u32,
    pub usage: vk::ImageUsageFlags,
    pub allocation_info: AllocationInfo,
    pub debug_name: Cow<'static, str>,
}

impl Default for ImageInfo {
    fn default() -> Self {
        Self {
            dimensions: 2,
            format: vk::Format::R8G8B8A8_UNORM,
            aspect: vk::ImageAspectFlags::COLOR,
            size: vk::Extent3D::default(),
            mip_level_count: 1,
            array_layer_count: 1,
            sample_count: 1,
            usage: vk::ImageUsageFlags::default(),
            allocation_info: AllocationInfo::Automatic(MemoryLocation::GpuOnly),
            debug_name: "".into()
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImageViewInfo {
    pub image_view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub image: ImageId,
    pub subresource_range: vk::ImageSubresourceRange,
    pub debug_name: Cow<'static, str>,
}

impl Default for ImageViewInfo {
    fn default() -> Self {
        Self {
            image_view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            image: ImageId::default(),
            subresource_range: vk::ImageSubresourceRange::default(),
            debug_name: "".into()
        }
    }
}

#[derive(Clone, Debug)]
pub struct SamplerInfo {
    pub magnification_filter: vk::Filter,
    pub minification_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub reduction_mode: vk::SamplerReductionMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub enable_anisotropy: bool,
    pub max_anisotropy: f32,
    pub enable_compare: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: vk::BorderColor,
    pub enable_unnormalized_coordinates: bool,
    pub debug_name: Cow<'static, str>,
}

impl Default for SamplerInfo {
    fn default() -> Self {
        Self {
            magnification_filter: vk::Filter::LINEAR,
            minification_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            reduction_mode: vk::SamplerReductionMode::WEIGHTED_AVERAGE,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            mip_lod_bias: 0.5,
            enable_anisotropy: false,
            max_anisotropy: 0.0,
            enable_compare: false,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: 1.0,
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            enable_unnormalized_coordinates: false,
            debug_name: "".into(),
        }
    }
}