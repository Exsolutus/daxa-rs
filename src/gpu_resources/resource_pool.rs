use crate::core::*;
use super::{
    GPUResourceId,
    BufferInfo,
    ImageInfo,
    ImageViewInfo,
    SamplerInfo
};

use ash::vk;
use gpu_allocator::vulkan::*;
use std::{
    cell::UnsafeCell,
    sync::Mutex,
};



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
pub(super) const PAGE_SIZE: usize = 1 << PAGE_BITS;
const PAGE_MASK: usize = PAGE_SIZE - 1;
pub(super) const PAGE_COUNT: usize = MAX_RESOURCE_COUNT / PAGE_SIZE;

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
    pub max_resources: usize,
    #[cfg(feature = "gpu_id_validation")]
    pub(super) use_after_free_check_mtx: Mutex<()>,
    pub(super) page_alloc_mtx: Mutex<IndexStack>,
    pub(super) pages: UnsafeCell<[Option<Box<[(ResourceT, u8); PAGE_SIZE]>>; PAGE_COUNT]>
}

pub(crate) struct IndexStack {
    pub free_index_stack: Vec<u32>,
    pub next_index: u32,
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
            max_resources: 0,
            #[cfg(feature = "gpu_id_validation")]
            use_after_free_check_mtx: Mutex::default(),
            page_alloc_mtx: Mutex::new(IndexStack {
                free_index_stack: vec![],
                next_index: 0,
            }),
            pages: UnsafeCell::new(pages)
        }
    }
}

impl<ResourceT: Slot + Default + std::fmt::Debug> GPUShaderResourcePool<ResourceT> {
    #[cfg(feature = "gpu_id_validation")]
    fn verify_resource_id(&self, id: &dyn ResourceId) {
        let page = (id.index() >> PAGE_BITS) as usize;
        let pages = unsafe { self.pages.get().as_ref().unwrap() };
        debug_assert!(page < pages.len(), "Detected invalid resource id.");
        debug_assert!(pages[page].is_some(), "Detected invalid resource id.");
        debug_assert!(id.version() != 0, "Detected invalid resource id.");
    }

    pub fn new_slot(&self) -> (GPUResourceId, &mut ResourceT) {
        #[cfg(feature = "gpu_id_validation")]
        let _use_after_free_check_lock = self.use_after_free_check_mtx.lock().unwrap();

        // Lock to ensure we have thread safe access to the pool
        let mut page_alloc_lock = self.page_alloc_mtx.lock().unwrap();
        let pages = unsafe { self.pages.get().as_mut().unwrap() };

        let index = match page_alloc_lock.free_index_stack.is_empty() {
            true => {
                let i = page_alloc_lock.next_index;
                page_alloc_lock.next_index += 1;
                debug_assert!((i as usize) < MAX_RESOURCE_COUNT, "Exceeded max resource count.");
                debug_assert!((i as usize) < self.max_resources, "Exceeded max resource count.");
                i
            },
            false => {
                page_alloc_lock.free_index_stack.pop().unwrap()
            }
        };

        let page = (index >> PAGE_BITS) as usize;
        let offset = (index & PAGE_MASK as u32) as usize;

        if pages[page].is_none() {
            let arr: Box<[(ResourceT, u8); PAGE_SIZE]> = (0..PAGE_SIZE) // TODO: maybe use a crate for array init
                .map(|_| {
                    (ResourceT::default(), 0u8)
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            pages[page] = Some(arr);
        }

        // Make sure the version is at least 1
        let (resource, version) = pages[page].as_mut().unwrap().get_mut(offset).unwrap();
        *version = (*version).max(1);

        let mut id = GPUResourceId(0);
        id.set_index(index);
        id.set_version(*version);

        (id, resource)
    }

    pub fn return_slot(&self, id: &dyn ResourceId) {
        let page = (id.index() >> PAGE_BITS) as usize;
        let offset = (id.index() & PAGE_MASK as u32) as usize;

        #[cfg(feature = "gpu_id_validation")]
        {
            let _use_after_free_check_lock = self.use_after_free_check_mtx.lock().unwrap();
            self.verify_resource_id(id);
            let pages = unsafe { self.pages.get().as_ref().unwrap() };
            debug_assert!(
                pages[page].as_ref().unwrap()[offset].1 == id.version(),
                "Detected double delete for a resource id."
            );
        }

        let mut page_alloc_mtx = self.page_alloc_mtx.lock().unwrap();
        let pages = unsafe { self.pages.get().as_mut().unwrap() };

        // Increment version
        let (_, version) = pages[page].as_mut().unwrap().get_mut(offset).unwrap();
        *version = (*version + 1).max(1); // the max is needed, as version = 0 is invalid

        page_alloc_mtx.free_index_stack.push(id.index());
    }

    pub fn is_id_valid(&self, id: &dyn ResourceId) -> bool {
        let page = (id.index() >> PAGE_BITS) as usize;
        let offset = (id.index() & PAGE_MASK as u32) as usize;

        let pages = unsafe { self.pages.get().as_ref().unwrap() };
        if page >= pages.len() || pages[page].is_none() || id.version() == 0 {
            return false
        }
        let version = pages[page].as_ref().unwrap()[offset].1;
        if version != id.version() || pages[page].as_ref().unwrap()[offset].0.is_zombie() {
            return false
        }

        true
    }

    pub fn dereference_id(&self, id: &dyn ResourceId) -> &ResourceT {
        let page = (id.index() >> PAGE_BITS) as usize;
        let offset = (id.index() & PAGE_MASK as u32) as usize;

        #[cfg(feature = "gpu_id_validation")]
        {
            self.verify_resource_id(id);
            let _use_after_free_check_lock = self.use_after_free_check_mtx.lock().unwrap();
            let pages = unsafe { self.pages.get().as_ref().unwrap() };
            let version = pages[page].as_ref().unwrap()[offset].1;
            debug_assert!(version != id.version(), "Detected use after free for a resource id.");
        }

        let pages = unsafe { self.pages.get().as_ref().unwrap() };

        &pages[page].as_ref().unwrap()[offset].0
    }

    pub fn dereference_id_mut(&self, id: &dyn ResourceId) -> &mut ResourceT {
        let page = (id.index() >> PAGE_BITS) as usize;
        let offset = (id.index() & PAGE_MASK as u32) as usize;

        #[cfg(feature = "gpu_id_validation")]
        {
            self.verify_resource_id(id);
            let _use_after_free_check_lock = self.use_after_free_check_mtx.lock().unwrap();
            let pages = unsafe { self.pages.get().as_ref().unwrap() };
            let version = pages[page].as_ref().unwrap()[offset].1;
            debug_assert!(version == id.version(), "Detected use after free for a resource id.");
        }

        let pages = unsafe { self.pages.get().as_mut().unwrap() };

        &mut pages[page].as_mut().unwrap()[offset].0
    }
}
