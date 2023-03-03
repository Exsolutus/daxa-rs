use crate::{device::*};

use anyhow::{Context, Result};
use ash::vk;

use std::{
    borrow::Cow,
    ffi::CStr,
    sync::{
        Arc,
        atomic::Ordering
    }
};



#[derive(Default)]
pub struct TimelineQueryPoolInfo {
    pub query_count: u32,
    pub debug_name: Cow<'static, str>,
}


#[derive(Default)]
pub(crate) struct TimelineQueryPoolZombie {
    pub timeline_query_pool: vk::QueryPool
}


#[derive(Clone)]
pub struct TimelineQueryPool(pub(crate) Arc<TimelineQueryPoolInternal>);

pub(crate) struct TimelineQueryPoolInternal {
    device: Device,
    pub info: TimelineQueryPoolInfo,
    pub timeline_query_pool: vk::QueryPool
}

// TimelineQueryPool creation methods
impl TimelineQueryPool {
    pub(crate) fn new(device: Device, info: TimelineQueryPoolInfo) -> Result<Self> {
        let query_pool_ci = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(info.query_count);

        let timeline_query_pool = unsafe {
            device.0.logical_device.create_query_pool(&query_pool_ci, None)
                .context("QueryPool should be created.")?
        };
        unsafe {
            device.0.logical_device.reset_query_pool(timeline_query_pool, 0, info.query_count);
        }

        #[cfg(debug_assertions)]
        unsafe {
            let query_pool_name = format!("{} [Daxa QueryPool]\0", info.debug_name);
            let query_pool_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::QUERY_POOL)
                .object_handle(vk::Handle::as_raw(timeline_query_pool))
                .object_name(&CStr::from_ptr(query_pool_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &query_pool_name_info)?;
        }

        Ok(Self(Arc::new(TimelineQueryPoolInternal {
            device,
            info,
            timeline_query_pool
        })))
    }
}

// TimelineQueryPool usage methods
impl TimelineQueryPool {
    pub fn info(&self) -> &TimelineQueryPoolInfo {
        &self.0.info
    }

    pub fn get_query_results(&self, start_index: u32, count: u32) -> Result<Vec<(u64, u64)>> {
        let internal = &self.0;

        debug_assert!(
            start_index + count - 1 < internal.info.query_count,
            "Attempt to query out of bounds for pool."
        );

        let mut results= vec![(0u64, 0u64); count as usize];
        unsafe {
            self.0.device.0.logical_device.get_query_pool_results(
                internal.timeline_query_pool,
                start_index,
                count,
                &mut results[..],
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY
            )
            .context("Query pool results should be acquired.")?
        };

        Ok(results)
    }
}

// TimelineQueryPool internal methods
impl TimelineQueryPoolInternal {

}

impl Drop for TimelineQueryPoolInternal {
    fn drop(&mut self) {
        let mut lock = self.device.0.main_queue_zombies.lock().unwrap();
        let cpu_timeline = self.device.0.main_queue_cpu_timeline.load(Ordering::Acquire);

        lock.timeline_query_pools.push_back(
            (cpu_timeline, TimelineQueryPoolZombie { timeline_query_pool: self.timeline_query_pool })
        );
    }
}