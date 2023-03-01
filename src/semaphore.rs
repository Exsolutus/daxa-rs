use crate::{device::Device};

use anyhow::{Result, Context};
use ash::vk;
use std::{
    ffi::{
        CStr
    },
    slice,
    sync::{
        Arc,
        atomic::Ordering
    }
};



#[derive(Clone, Copy, Default)]
pub struct BinarySemaphoreInfo {
    pub debug_name: &'static str,
}

#[derive(Clone, Copy, Default)]
pub struct TimelineSemaphoreInfo {
    pub initial_value: u64,
    pub debug_name: &'static str,
}


pub(crate) struct SemaphoreZombie {
    pub semaphore: vk::Semaphore
}


#[derive(Clone)]
pub struct BinarySemaphore(pub(crate) Arc<BinarySemaphoreInternal>);

pub(crate) struct BinarySemaphoreInternal {
    device: Device,
    pub semaphore: vk::Semaphore,
    info: BinarySemaphoreInfo,
}

// BinarySemaphore creation methods
impl BinarySemaphore {
    pub fn new(
        device: Device,
        info: BinarySemaphoreInfo
    ) -> Result<Self> {
        let semaphore_ci = vk::SemaphoreCreateInfo::builder();

        let semaphore = unsafe {
            device.0.logical_device.create_semaphore(&semaphore_ci, None)
                .context("Device should create a semaphore.")?
        };

        #[cfg(debug_assertions)]
        unsafe {
            let semaphore_name = format!("{} [Daxa BinarySemaphore]\0", info.debug_name);
            let semaphore_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::SEMAPHORE)
                .object_handle(vk::Handle::as_raw(semaphore))
                .object_name(&CStr::from_ptr(semaphore_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &semaphore_name_info)?;
        }

        Ok(Self(Arc::new(BinarySemaphoreInternal {
            device,
            semaphore,
            info
        })))
    }
}

// BinarySemaphore usage methods
impl BinarySemaphore {
    #[inline]
    pub fn info(&self) -> &BinarySemaphoreInfo {
        &self.0.info
    }
}

// BinarySemaphore internal methods
impl BinarySemaphoreInternal {

}

impl Drop for BinarySemaphoreInternal {
    fn drop(&mut self) {
        let timeline = self.device.0.main_queue_cpu_timeline.load(Ordering::Acquire);

        self.device.0.main_queue_zombies.lock()
            .unwrap()
            .semaphores.push_back((
                timeline,
                SemaphoreZombie {
                    semaphore: self.semaphore
                }
            ));
    }
}



#[derive(Clone)]
pub struct TimelineSemaphore(pub(crate) Arc<TimelineSemaphoreInternal>);

pub(crate) struct TimelineSemaphoreInternal {
    device: Device,
    pub semaphore: vk::Semaphore,
    info: TimelineSemaphoreInfo
}


// TimelineSemaphore creation methods
impl TimelineSemaphore {
    pub fn new(
        device: Device,
        info: TimelineSemaphoreInfo
    ) -> Result<Self> {
        let mut semaphore_type_ci = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(info.initial_value)
            .build();

        let semaphore_ci = vk::SemaphoreCreateInfo::builder()
            .push_next(&mut semaphore_type_ci);
        
        let semaphore = unsafe {
            device.0.logical_device.create_semaphore(&semaphore_ci, None)
                .context("Device should create a semaphore.")?
        };

        #[cfg(debug_assertions)]
        unsafe {
            let semaphore_name = format!("{} [Daxa TimelineSemaphore]\0", info.debug_name);
            let semaphore_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::SEMAPHORE)
                .object_handle(vk::Handle::as_raw(semaphore))
                .object_name(&CStr::from_ptr(semaphore_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &semaphore_name_info)?;
        }

        Ok(Self(Arc::new(TimelineSemaphoreInternal {
            device,
            semaphore,
            info
        })))
    }
}

// TimelineSemaphore usage methods
impl TimelineSemaphore {
    #[inline]
    pub fn info(&self) -> &TimelineSemaphoreInfo {
        &self.0.info
    }

    #[inline]
    pub fn value(&self) -> u64 {
        unsafe {
            // TODO: maybe handle errors here better
            self.0.device.0.logical_device.get_semaphore_counter_value(self.0.semaphore).unwrap()
        }
    }

    pub fn set_value(&self, value: u64) {
        let signal_info = vk::SemaphoreSignalInfo::builder()
            .semaphore(self.0.semaphore)
            .value(value);

        unsafe { self.0.device.0.logical_device.signal_semaphore(&signal_info).unwrap_unchecked() };
    }

    pub fn wait_for_value(
        &self,
        value: u64,
        timeout: u64
    ) -> bool {
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(slice::from_ref(&self.0.semaphore))
            .values(slice::from_ref(&value));

        unsafe { self.0.device.0.logical_device.wait_semaphores(&wait_info, timeout) != Err(vk::Result::TIMEOUT) }
    }
}

// TimelineSemaphore internal methods
impl TimelineSemaphoreInternal {

}

impl Drop for TimelineSemaphoreInternal {
    fn drop(&mut self) {
        let timeline = self.device.0.main_queue_cpu_timeline.load(Ordering::Acquire);

        self.device.0.main_queue_zombies.lock()
            .unwrap()
            .semaphores.push_back((
                timeline,
                SemaphoreZombie {
                    semaphore: self.semaphore
                }
            ));
    }
}