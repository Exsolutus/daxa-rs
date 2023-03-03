use crate::core::*;
use super::{
    resource_pool::*,
    BUFFER_BINDING,
    STORAGE_IMAGE_BINDING,
    SAMPLED_IMAGE_BINDING,
    SAMPLER_BINDING,
    BUFFER_DEVICE_ADDRESS_BUFFER_BINDING
};

use anyhow::{Result, Context};
use ash::vk;
use std::{
    slice,
    any::type_name,
    mem::transmute_copy
};



pub(crate) struct GPUShaderResourceTable {
    pub buffer_slots: GPUShaderResourcePool<BufferSlot>,
    pub image_slots: GPUShaderResourcePool<ImageSlot>,
    pub sampler_slots: GPUShaderResourcePool<SamplerSlot>,

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
        max_timeline_query_pools: usize,
        device: &ash::Device,
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

    pub fn cleanup(&self, device: &ash::Device) {
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
                        let mut debug_name = "".into();
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
            
            #[cfg(debug_assertions)]
            {
                let buffer_lock = self.buffer_slots.page_alloc_mtx.lock().unwrap();
                let image_lock = self.image_slots.page_alloc_mtx.lock().unwrap();
                let sampler_lock = self.sampler_slots.page_alloc_mtx.lock().unwrap();

                let buffer_pages = unsafe { self.buffer_slots.pages.get().as_ref().unwrap() };
                let image_pages = unsafe { self.image_slots.pages.get().as_ref().unwrap() };
                let sampler_pages = unsafe { self.sampler_slots.pages.get().as_ref().unwrap() };

                debug_assert!(buffer_lock.free_index_stack.len() == buffer_lock.next_index as usize, "{}", print_remaining("Not all buffers have been destroyed before destroying the device.", buffer_pages));
                debug_assert!(image_lock.free_index_stack.len() == image_lock.next_index as usize, "{}", print_remaining("Not all images have been destroyed before destroying the device.", image_pages));
                debug_assert!(sampler_lock.free_index_stack.len() == sampler_lock.next_index as usize, "{}", print_remaining("Not all samplers have been destroyed before destroying the device.", sampler_pages));
            }
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

    pub fn write_descriptor_set_buffer(&self, device: &ash::Device, buffer: vk::Buffer, offset: vk::DeviceSize, range: vk::DeviceSize, index: u32) {
        let descriptor_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(buffer)
            .offset(offset)
            .range(range)
            .build();

        let write_descriptor_set = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(BUFFER_BINDING)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(slice::from_ref(&descriptor_buffer_info))
            .build();

        unsafe {
            device.update_descriptor_sets(slice::from_ref(&write_descriptor_set), &[]);
        }
    }

    pub fn write_descriptor_set_image(&self, device: &ash::Device, image_view: vk::ImageView, usage: vk::ImageUsageFlags, index: u32) {
        let mut descriptor_set_writes = vec![];
        
        if usage.contains(vk::ImageUsageFlags::STORAGE) {
            let descriptor_image_info = vk::DescriptorImageInfo::builder()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::GENERAL)
                .build();

            let write_descriptor_set = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(STORAGE_IMAGE_BINDING)
                .dst_array_element(index)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(slice::from_ref(&descriptor_image_info))
                .build();

            descriptor_set_writes.push(write_descriptor_set);
        }

        if usage.contains(vk::ImageUsageFlags::SAMPLED) {
            let descriptor_image_info = vk::DescriptorImageInfo::builder()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .build();

            let write_descriptor_set = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(SAMPLED_IMAGE_BINDING)
                .dst_array_element(index)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(slice::from_ref(&descriptor_image_info))
                .build();

            descriptor_set_writes.push(write_descriptor_set);
        }

        unsafe {
            device.update_descriptor_sets(descriptor_set_writes.as_slice(), &[]);
        }
    }

    pub fn write_descriptor_set_sampler(&self, device: &ash::Device, sampler: vk::Sampler, index: u32) {
        let descriptor_image_info = vk::DescriptorImageInfo::builder()
            .sampler(sampler)
            .build();

        let write_descriptor_set = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(SAMPLER_BINDING)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(slice::from_ref(&descriptor_image_info))
            .build();

        unsafe {
            device.update_descriptor_sets(slice::from_ref(&write_descriptor_set), &[]);
        }
    }
}