use crate::core::*;

use anyhow::{Result, Context};
use ash::vk;
use bitfield::*;
use gpu_allocator::vulkan::*;
use std::{
    ffi::{
        c_void
    },
    fmt::Display,
    slice,
    sync::{
        Mutex
    },
    any::type_name, mem::{transmute, transmute_copy}, ops::Add
};



pub const BUFFER_BINDING: u32 = 0;
pub const STORAGE_IMAGE_BINDING: u32 = 1;
pub const SAMPLED_IMAGE_BINDING: u32 = 2;
pub const SAMPLER_BINDING: u32 = 3;
pub const BUFFER_DEVICE_ADDRESS_BUFFER_BINDING: u32 = 4;

pub const NOT_OWNED_BY_SWAPCHAIN: i32 = -1;



#[derive(ResourceId, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct GPUResourceId(pub u32);

#[derive(ResourceId, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct BufferId(pub u32);

impl Display for BufferId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "index: {}\nversion: {}", self.index(), self.version())
    }
}

#[derive(ResourceId, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct ImageId(pub u32);

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
pub struct ImageViewId(pub u32);

#[derive(ResourceId, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct SamplerId(pub u32);



#[derive(Clone, Copy, Debug, Default)]
pub struct BufferInfo {
    pub memory_flags: vk::MemoryPropertyFlags,
    pub size: u32,
    pub debug_name: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct ImageInfo {
    pub dimensions: u32,
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
    pub size: vk::Extent3D,
    pub mip_level_count: u32,
    pub array_layer_count: u32,
    pub sample_count: u32,
    pub usage: vk::ImageUsageFlags,
    pub memory_flags: vk::MemoryPropertyFlags,
    pub debug_name: &'static str,
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
            memory_flags: vk::MemoryPropertyFlags::default(),
            debug_name: ""
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageViewInfo {
    pub image_view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub image: ImageId,
    pub subresource_range: vk::ImageSubresourceRange,
    pub debug_name: &'static str,
}

impl Default for ImageViewInfo {
    fn default() -> Self {
        Self {
            image_view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            image: ImageId::default(),
            subresource_range: vk::ImageSubresourceRange::default(),
            debug_name: ""
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SamplerInfo {
    pub magnification_filter: vk::Filter,
    pub minification_filder: vk::Filter,
    pub mipmap_filter: vk::Filter,
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
    pub debug_name: &'static str,
}

impl Default for SamplerInfo {
    fn default() -> Self {
        Self {
            magnification_filter: vk::Filter::LINEAR,
            minification_filder: vk::Filter::LINEAR,
            mipmap_filter: vk::Filter::LINEAR,
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
            debug_name: "",
        }
    }
}




#[derive(Slot, Debug, Default)]
pub(crate) struct BufferSlot {
    pub info: BufferInfo,
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub device_address: vk::DeviceAddress,
    // host_address: use allocation.mapped_ptr().
    pub zombie: bool,
}

#[derive(Slot, Debug, Default)]
pub(crate) struct ImageViewSlot {
    pub info: ImageViewInfo,
    pub image_view: vk::ImageView,
    pub zombie: bool,
}

#[derive(Slot, Debug, Default)]
pub(crate) struct ImageSlot {
    pub view_slot: ImageViewSlot,
    pub info: ImageInfo,
    pub image: vk::Image,
    pub allocation: Allocation,
    pub swapchain_image_index: i32,
    pub zombie: bool
}

#[derive(Slot, Debug, Default)]
pub(crate) struct SamplerSlot {
    pub info: SamplerInfo,
    pub sampler: vk::Sampler,
    pub zombie: bool
}



const MAX_RESOURCE_COUNT: usize = 1 << 20;
const PAGE_BITS: usize = 12;
const PAGE_SIZE: usize = 1 << PAGE_BITS;
const PAGE_MASK: usize = PAGE_SIZE - 1;
const PAGE_COUNT: usize = MAX_RESOURCE_COUNT / PAGE_SIZE;

/// [`GPUShaderResourcePool`] is intended to be used akin to a specialized memory allocator, specific to gpu resource types (like image views).
/// 
/// This struct is threadsafe if the following assumptions are met:
/// * never dereference a deleted resource
/// * never delete a resource twice
/// That means the function dereference_id can be used without synchronization, even calling get_new_slot or return_old_slot in parallel is safe.
/// 
/// To check if these assumptions are met at runtime, the debug define DAXA_GPU_ID_VALIDATION can be enabled.
/// The define enables runtime checking to detect use after free and double free at the cost of performance.
pub(crate) struct GPUShaderResourcePool<ResourceT: Slot + Default + std::fmt::Debug> {
    free_index_stack: Vec<u32>,
    next_index: u32,
    max_resources: usize,
    #[cfg(feature = "gpu_id_validation")]
    use_after_free_check_mtx: Mutex<()>,
    page_alloc_mtx: Mutex<()>,
    pages: [Option<Box<[(ResourceT, u8); PAGE_SIZE]>>; PAGE_COUNT],
}

impl<ResourceT: Slot + Default + std::fmt::Debug> Default for GPUShaderResourcePool<ResourceT> {
    fn default() -> Self {
        let pages = (0..PAGE_COUNT) // TODO: maybe use a crate for array init
            .map(|_| {
                None
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        
        Self {
            free_index_stack: vec![],
            next_index: 0,
            max_resources: 0,
            #[cfg(feature = "gpu_id_validation")]
            use_after_free_check_mtx: Mutex::default(),
            page_alloc_mtx: Mutex::default(),
            pages,
        }
    }
}

impl<ResourceT: Slot + Default + std::fmt::Debug> GPUShaderResourcePool<ResourceT> {
    #[cfg(feature = "gpu_id_validation")]
    fn verify_resource_id(&mut self, id: &dyn ResourceId) {
        let page = (id.index() >> PAGE_BITS) as usize;

        debug_assert!(page >= self.pages.len(), "Detected invalid resource id.");
        debug_assert!(!self.pages[page].is_none(), "Detected invalid resource id.");
        debug_assert!(id.version() == 0, "Detected invalid resource id.");
    }

    pub fn new_slot(&mut self) -> (GPUResourceId, &ResourceT) {
        #[cfg(feature = "gpu_id_validation")]
        let use_after_free_check_lock = self.use_after_free_check_mtx.lock().unwrap();

        let page_alloc_lock = self.page_alloc_mtx.lock().unwrap();

        let index = match self.free_index_stack.is_empty() {
            true => {
                let i = self.next_index;
                self.next_index += 1;
                debug_assert!(i >= MAX_RESOURCE_COUNT as u32, "Exceeded max resource count.");
                debug_assert!(i >= self.max_resources as u32, "Exceeded max resource count.");
                i
            },
            false => {
                self.free_index_stack.pop().unwrap()
            }
        };

        let page = (index >> PAGE_BITS) as usize;
        let offset = (index & PAGE_MASK as u32) as usize;

        if self.pages[page].is_none() {
            let arr = (0..PAGE_SIZE) // TODO: maybe use a crate for array init
                .map(|_| {
                    (ResourceT::default(), 0u8)
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            self.pages[page] = Some(Box::new(arr));
        }

        // Make sure the version is at least 1
        let (resource, version) = self.pages[page].as_mut().unwrap().get_mut(offset).unwrap();
        *version = (*version).max(1);

        let mut id = GPUResourceId(0);
        id.set_index(index);
        id.set_version(*version);

        (id, resource)
    }

    pub fn return_slot(&mut self, id: &dyn ResourceId) {
        let page = (id.index() >> PAGE_BITS) as usize;
        let offset = (id.index() & PAGE_MASK as u32) as usize;

        #[cfg(feature = "gpu_id_validation")]
        {
            self.verify_resource_id(id);
            let use_after_free_check_lock = self.use_after_free_check_mtx.lock().unwrap();
            debug_assert!(self.pages[page].as_ref().unwrap()[offset].1 != id.version(), "Detected double delete for a resource id.");
        }

        let page_alloc_lock = self.page_alloc_mtx.lock().unwrap();

        // Increment version
        let (_, version) = self.pages[page].as_mut().unwrap().get_mut(offset).unwrap();
        *version = (*version + 1).max(1); // the max is needed, as version = 0 is invalid

        self.free_index_stack.push(id.index());
    }

    pub fn is_id_valid(&self, id: &dyn ResourceId) -> bool {
        let page = (id.index() >> PAGE_BITS) as usize;
        let offset = (id.index() & PAGE_MASK as u32) as usize;

        if page >= self.pages.len() || self.pages[page].is_none() || id.version() == 0 {
            return false
        }
        let version = self.pages[page].as_ref().unwrap()[offset].1;
        if version != id.version() || self.pages[page].as_ref().unwrap()[offset].0.is_zombie() {
            return false
        }

        true
    }

    pub fn dereference_id(&mut self, id: &dyn ResourceId) -> &ResourceT {
        let page = (id.index() >> PAGE_BITS) as usize;
        let offset = (id.index() & PAGE_MASK as u32) as usize;

        #[cfg(feature = "gpu_id_validation")]
        {
            self.verify_resource_id(id);
            let use_after_free_check_lock = self.use_after_free_check_mtx.lock().unwrap();
            let version = self.pages[page].as_ref().unwrap()[offset].1;
            debug_assert!(version != id.version(), "Detected use after free for a resource id.");
        }

        &self.pages[page].as_ref().unwrap()[offset].0
    }
}

pub(crate) struct GPUShaderResourceTable {
    buffer_slots: GPUShaderResourcePool<BufferSlot>,
    image_slots: GPUShaderResourcePool<ImageSlot>,
    sampler_slots: GPUShaderResourcePool<SamplerSlot>,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,

    // Contains pipeline layouts with varying push constant range size.
    // The first size is 0 word, second is 1 word, all others are a power of two (maximum is MAX_PUSH_CONSTANT_BYTE_SIZE).
    pipeline_layouts: [vk::PipelineLayout; PIPELINE_LAYOUT_COUNT as usize],
}

impl GPUShaderResourceTable {
    pub fn new(
        max_buffers: usize,
        max_images: usize,
        max_samplers: usize,
        max_timeline_query_pool: usize,
        device: ash::Device,
        device_address_buffer: vk::Buffer
    ) -> Result<Self> {
        let buffer_slots = GPUShaderResourcePool {
            max_resources: max_buffers,
            ..Default::default()
        };
        let image_slots = GPUShaderResourcePool {
            max_resources: max_images,
            ..Default::default()
        };
        let sampler_slots = GPUShaderResourcePool {
            max_resources: max_samplers,
            ..Default::default()
        };

        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .descriptor_count((buffer_slots.max_resources + 1) as u32)
                .build(),
            vk::DescriptorPoolSize::builder()
                .descriptor_count(image_slots.max_resources as u32)
                .build(),
            vk::DescriptorPoolSize::builder()
                .descriptor_count(image_slots.max_resources as u32)
                .build(),
            vk::DescriptorPoolSize::builder()
                .descriptor_count(sampler_slots.max_resources as u32)
                .build(),
        ];

        let descriptor_pool_ci = vk::DescriptorPoolCreateInfo::builder()
            .flags(
                vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET |
                vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
            )
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&descriptor_pool_ci, None)
                .context("DescriptorPool should be created.")?
        };

        let descriptor_set_layout_bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(buffer_slots.max_resources as u32)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(STORAGE_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(image_slots.max_resources as u32)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(SAMPLED_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(image_slots.max_resources as u32)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(SAMPLER_BINDING)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(sampler_slots.max_resources as u32)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(BUFFER_DEVICE_ADDRESS_BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
        ];

        let descriptor_binding_flags = [
            vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING | vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING | vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING | vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING | vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING | vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];

        let mut descriptor_set_layout_binding_flags_ci = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
            .binding_flags(&descriptor_binding_flags)
            .build();

        let descriptor_set_layout_ci = vk::DescriptorSetLayoutCreateInfo::builder()
            .push_next(&mut descriptor_set_layout_binding_flags_ci)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&descriptor_set_layout_bindings)
            .build();

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_ci, None)
                .context("DescriptorSetLayout should be created.")?
        };

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(slice::from_ref(&descriptor_set_layout))
            .build();

        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(&descriptor_set_allocate_info)
                .context("DescriptorSet should be allocated")?[0]
        };

        let mut pipeline_ci = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(slice::from_ref(&descriptor_set_layout))
            .build();

        let mut pipeline_layouts = vec![
            unsafe { device.create_pipeline_layout(&pipeline_ci, None).context("PipelineLayout should be created.")? }
        ];

        for i in 1..PIPELINE_LAYOUT_COUNT {
            let push_constant_range = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::ALL)
                .size(i * 4)
                .build();
            pipeline_ci.push_constant_range_count = 1;
            pipeline_ci.p_push_constant_ranges = &push_constant_range;
            pipeline_layouts.push(
                unsafe { device.create_pipeline_layout(&pipeline_ci, None).context("PipelineLayout should be created.")? }
            )
        }

        let write_buffer = vk::DescriptorBufferInfo::builder()
            .buffer(device_address_buffer)
            .range(vk::WHOLE_SIZE)
            .build();

        let write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(BUFFER_DEVICE_ADDRESS_BUFFER_BINDING)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(slice::from_ref(&write_buffer))
            .build();

        unsafe {
            device.update_descriptor_sets(slice::from_ref(&write), &[]);
        }

        Ok(Self {
            buffer_slots,
            image_slots,
            sampler_slots,
            descriptor_set_layout,
            descriptor_set,
            descriptor_pool,
            pipeline_layouts: pipeline_layouts.try_into().unwrap()
        })
    }

    pub fn cleanup(&self, device: ash::Device) {
        #[cfg(debug_assertions)]
        {
            fn print_remaining<T>(prefix: &str, pages: &[Option<Box<[(T, u8); PAGE_SIZE]>>; PAGE_COUNT]) -> String {
                let mut ret = format!("{}\nThis can happen due to not waiting for the gpu to finish executing, as daxa deferres destruction. List of survivors:\n", prefix);
                for page in pages {
                    if page.is_none() {
                        continue;
                    }

                    for slot in page.as_ref().unwrap().iter() {
                        let mut handle_invalid = false;
                        let mut debug_name = "";
                        let mut zombie = false;

                        if type_name::<T>() == type_name::<BufferSlot>() {
                            let slot = unsafe { transmute_copy::<T, BufferSlot>(&slot.0) };
                            handle_invalid = slot.buffer == vk::Buffer::null();
                            debug_name = slot.info.debug_name;
                            zombie = slot.zombie;
                        }
                        if type_name::<T>() == type_name::<ImageSlot>() {
                            let slot = unsafe { transmute_copy::<T, ImageSlot>(&slot.0) };
                            handle_invalid = slot.image == vk::Image::null();
                            debug_name = slot.info.debug_name;
                            zombie = slot.zombie;
                        }
                        if type_name::<T>() == type_name::<SamplerSlot>() {
                            let slot = unsafe { transmute_copy::<T, SamplerSlot>(&slot.0) };
                            handle_invalid = slot.sampler == vk::Sampler::null();
                            debug_name = slot.info.debug_name;
                            zombie = slot.zombie;
                        }
                        if !handle_invalid {
                            ret += format!("debug name: \"{}\"", debug_name).as_str();
                            if zombie {
                                ret += " (destroy was already called)";
                            }
                            ret += "\n";
                        }
                    }
                }
                ret
            }
            
            debug_assert!(self.buffer_slots.free_index_stack.len() == self.buffer_slots.next_index as usize, "{}", print_remaining("Not all buffers have been destroyed before destroying the device.", &self.buffer_slots.pages));
            debug_assert!(self.image_slots.free_index_stack.len() == self.image_slots.next_index as usize, "{}", print_remaining("Not all images have been destroyed before destroying the device.", &self.image_slots.pages));
            debug_assert!(self.sampler_slots.free_index_stack.len() == self.sampler_slots.next_index as usize, "{}", print_remaining("Not all samplers have been destroyed before destroying the device.", &self.sampler_slots.pages));
        }

        unsafe {
            for layout in self.pipeline_layouts {
                device.destroy_pipeline_layout(layout, None);
            }
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }

    pub fn write_descriptor_set_buffer(device: ash::Device, buffer: vk::Buffer, offset: vk::DeviceSize, range: vk::DeviceSize, index: u32) {
        todo!()
    }

    pub fn write_descriptor_set_image(device: ash::Device, image: vk::Image, usage: vk::ImageUsageFlags, index: u32) {
        todo!()
    }

    pub fn write_descriptor_set_sampler(device: ash::Device, sampler: vk::Sampler, index: u32) {
        todo!()
    }
}