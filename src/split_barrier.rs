use crate::{types::*, device::*, gpu_resources::*};

use anyhow::{Context, Result};
use ash::vk;

use std::{
    borrow::Cow,
    ffi::CStr,
    sync::{
        atomic::Ordering
    }
};



#[derive(Default)]
pub struct MemoryBarrierInfo {
    pub src_access: Access,
    pub dst_access: Access
}


pub struct ImageBarrierInfo {
    pub src_access: Access,
    pub dst_access: Access,
    pub src_layout: vk::ImageLayout,
    pub dst_layout: vk::ImageLayout,
    pub range: vk::ImageSubresourceRange,
    pub image: ImageId
}

impl Default for ImageBarrierInfo {
    fn default() -> Self {
        Self {
            src_access: Default::default(),
            dst_access: Default::default(),
            src_layout: Default::default(),
            dst_layout: Default::default(),
            range: vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
            image: Default::default()
        }
    }
}


#[derive(Clone)]
pub struct SplitBarrierInfo {
    pub debug_name: Cow<'static, str>
}

pub struct SplitBarrierSignalInfo<'a> {
    pub memory_barriers: &'a [MemoryBarrierInfo],
    pub image_barriers: &'a [ImageBarrierInfo],
    pub split_barrier: SplitBarrierState
}

pub type SplitBarrierWaitInfo<'a> = SplitBarrierSignalInfo<'a>;



pub(crate) struct SplitBarrierZombie {
    pub event: vk::Event
}



#[derive(Clone)]
pub struct SplitBarrierState {
    device: Device,
    info: SplitBarrierInfo,
    data: u64
}

// SplitBarrierState creation methods
impl SplitBarrierState {
    pub(crate) fn new(device: Device, info: SplitBarrierInfo) -> Result<Self> {
        let event_ci = vk::EventCreateInfo::builder()
            .flags(vk::EventCreateFlags::DEVICE_ONLY);

        let event = unsafe {
            device.0.logical_device.create_event(&event_ci, None)
                .context("Event should be created.")?
        };

        #[cfg(debug_assertions)]
        unsafe {
            let event_name = format!("{} [Daxa SplitBarrier]\0", info.debug_name);
            let event_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::EVENT)
                .object_handle(vk::Handle::as_raw(event))
                .object_name(&CStr::from_ptr(event_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &event_name_info)?;
        }

        Ok(Self {
            device,
            info,
            data: vk::Handle::as_raw(event)
        })
    }
}

// SplitBarrierState usage methods
impl SplitBarrierState {
    pub fn info(&self) -> &SplitBarrierInfo {
        &self.info
    }
}

impl Drop for SplitBarrierState {
    fn drop(&mut self) {
        if self.data != 0 {
            let mut lock = self.device.0.main_queue_zombies.lock().unwrap();
            let cpu_timeline = self.device.0.main_queue_cpu_timeline.load(Ordering::Acquire);

            lock.split_barriers.push_back(
                (cpu_timeline, SplitBarrierZombie { event: vk::Handle::from_raw(self.data) })
            )
        }
    }
}