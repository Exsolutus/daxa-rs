use crate::{device::{Device, Zombie}};

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
impl Zombie for SemaphoreZombie { }

pub(crate) struct BinarySemaphoreInternal {
    device: Device,
    pub semaphore: vk::Semaphore,
    info: BinarySemaphoreInfo,
}

#[derive(Clone)]
pub struct BinarySemaphore {
    pub(crate) internal: Arc<BinarySemaphoreInternal>
}

// BinarySemaphore creation methods
impl BinarySemaphore {
    pub fn new(
        device: Device,
        info: BinarySemaphoreInfo
    ) -> Result<Self> {
        let semaphore_ci = vk::SemaphoreCreateInfo::builder();

        let semaphore = unsafe {
            device.create_semaphore(&semaphore_ci, None)
                .context("Device should create a semaphore.")?
        };

        #[cfg(debug_assertions)]
        unsafe {
            let semaphore_name = format!("{} [Daxa BinarySemaphore]\0", info.debug_name);
            let semaphore_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::SEMAPHORE)
                .object_handle(vk::Handle::as_raw(semaphore))
                .object_name(&CStr::from_ptr(semaphore_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.handle(), &semaphore_name_info)?;
        }

        Ok(Self {
            internal: Arc::new(BinarySemaphoreInternal {
                device,
                semaphore,
                info
            })
        })
    }
}

// BinarySemaphore usage methods
impl BinarySemaphore {
    #[inline]
    pub fn info(&self) -> &BinarySemaphoreInfo {
        &self.internal.info
    }
}

// BinarySemaphore internal methods
impl BinarySemaphoreInternal {

}

impl Drop for BinarySemaphoreInternal {
    fn drop(&mut self) {
        let timeline = self.device.internal.main_queue_cpu_timeline.load(Ordering::Acquire);

        self.device.internal.main_queue_zombies.lock()
            .unwrap()
            .semaphores.push_back((
                timeline,
                SemaphoreZombie {
                    semaphore: self.semaphore
                }
            ));
    }
}



pub(crate) struct TimelineSemaphoreInternal {
    device: Device,
    pub semaphore: vk::Semaphore,
    info: TimelineSemaphoreInfo
}

#[derive(Clone)]
pub struct TimelineSemaphore {
    pub(crate) internal: Arc<TimelineSemaphoreInternal>
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
            device.create_semaphore(&semaphore_ci, None)
                .context("Device should create a semaphore.")?
        };

        #[cfg(debug_assertions)]
        unsafe {
            let semaphore_name = format!("{} [Daxa TimelineSemaphore]\0", info.debug_name);
            let semaphore_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::SEMAPHORE)
                .object_handle(vk::Handle::as_raw(semaphore))
                .object_name(&CStr::from_ptr(semaphore_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.handle(), &semaphore_name_info)?;
        }

        Ok(Self {
            internal: Arc::new(TimelineSemaphoreInternal {
                device,
                semaphore,
                info
            })
        })
    }
}

// TimelineSemaphore usage methods
impl TimelineSemaphore {
    #[inline]
    pub fn info(&self) -> &TimelineSemaphoreInfo {
        &self.internal.info
    }

    #[inline]
    pub fn value(&self) -> u64 {
        unsafe {
            // TODO: maybe handle errors here better
            self.internal.device.get_semaphore_counter_value(self.internal.semaphore).unwrap()
        }
    }

    pub fn set_value(&self, value: u64) {
        let signal_info = vk::SemaphoreSignalInfo::builder()
            .semaphore(self.internal.semaphore)
            .value(value);

        unsafe { self.internal.device.signal_semaphore(&signal_info).unwrap_unchecked() };
    }

    pub fn wait_for_value(
        &self,
        value: u64,
        timeout: u64
    ) -> bool {
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(slice::from_ref(&self.internal.semaphore))
            .values(slice::from_ref(&value));

        unsafe { self.internal.device.wait_semaphores(&wait_info, timeout) != Err(vk::Result::TIMEOUT) }
    }
}

// TimelineSemaphore internal methods
impl TimelineSemaphoreInternal {

}

impl Drop for TimelineSemaphoreInternal {
    fn drop(&mut self) {
        let timeline = self.device.internal.main_queue_cpu_timeline.load(Ordering::Acquire);

        self.device.internal.main_queue_zombies.lock()
            .unwrap()
            .semaphores.push_back((
                timeline,
                SemaphoreZombie {
                    semaphore: self.semaphore
                }
            ));
    }
}



#[cfg(test)]
mod tests {
    use crate::{context::*, device::*, command_list::*};
    use super::{BinarySemaphoreInfo};

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
    fn binary_semaphore() {
        let app = App::new();

        let command_list1 = app.device.create_command_list(CommandListInfo::default()).unwrap();
        command_list1.complete();

        let command_list2 = app.device.create_command_list(CommandListInfo::default()).unwrap();
        command_list2.complete();

        let command_list3 = app.device.create_command_list(CommandListInfo::default()).unwrap();
        command_list3.complete();

        let command_list4 = app.device.create_command_list(CommandListInfo::default()).unwrap();
        command_list4.complete();

        let binary_semaphore1 = app.device.create_binary_semaphore(BinarySemaphoreInfo::default()).unwrap();
        let binary_semaphore2 = app.device.create_binary_semaphore(BinarySemaphoreInfo::default()).unwrap();

        // This semaphore is useful in the future, it can be used to make submits wait on each other,
        // or to make a present wait for a submit to finish
        app.device.submit_commands(CommandSubmitInfo {
            command_lists: vec![command_list1],
            signal_binary_semaphores: vec![binary_semaphore1.clone()],
            ..Default::default()
        });
        
        app.device.submit_commands(CommandSubmitInfo {
            command_lists: vec![command_list2],
            wait_binary_semaphores: vec![binary_semaphore1.clone()],
            signal_binary_semaphores: vec![binary_semaphore2.clone()],
            ..Default::default()
        });
        
        // Binary semaphores can be reused ONLY after they have been signaled.
        app.device.submit_commands(CommandSubmitInfo {
            command_lists: vec![command_list3],
            wait_binary_semaphores: vec![binary_semaphore2.clone()],
            signal_binary_semaphores: vec![binary_semaphore1.clone()],
            ..Default::default()
        });
        
        app.device.submit_commands(CommandSubmitInfo {
            command_lists: vec![command_list4],
            wait_binary_semaphores: vec![binary_semaphore1.clone()],
            ..Default::default()
        });
    }
}