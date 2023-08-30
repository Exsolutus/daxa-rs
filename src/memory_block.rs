use crate::{
    core::*,
    device::Device
};

use anyhow::Context;
// Reexport
pub use ash::vk::{
    MemoryRequirements,
};
pub use gpu_allocator::{
    MemoryLocation
};

use std::rc::Rc;



#[derive(Debug)]
pub struct MemoryBlockInfo {
    pub name: &'static str,
    pub requirements: MemoryRequirements,
    pub location: MemoryLocation
}

#[derive(Clone, Debug)]
pub enum AllocationInfo {
    Automatic(MemoryLocation),
    Manual {
        memory_block: MemoryBlock,
        offset: usize
    }
}

#[derive(Clone)]
pub struct MemoryBlock(pub(crate) Rc<MemoryBlockInternal>);

pub(crate) struct MemoryBlockInternal {
    device: Device,
    pub info: MemoryBlockInfo,
    pub allocation: gpu_allocator::vulkan::Allocation,
    allocation_info: gpu_allocator::vulkan::AllocationCreateDesc<'static>
}

impl MemoryBlock {
    pub fn new(
        device: Device,
        info: MemoryBlockInfo,
        allocation: gpu_allocator::vulkan::Allocation,
        allocation_info: gpu_allocator::vulkan::AllocationCreateDesc<'static>
    ) -> Self {
        Self(MemoryBlockInternal {
            device,
            info,
            allocation,
            allocation_info
        }.into())
    }
}

impl Drop for MemoryBlockInternal {
    fn drop(&mut self) {
        let allocation = std::mem::take(&mut self.allocation);
        self.device.0.allocator
            .lock()
            .unwrap()
            .free(allocation)
            .unwrap();
    }
}

impl std::fmt::Debug for MemoryBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryBlock")
            .field("info", &self.0.info)
            .field("allocation", &self.0.allocation)
            .field("allocation_info", &self.0.allocation_info)
            .finish()
    }
}


pub(crate) struct MemoryBlockZombie {
    allocation: gpu_allocator::vulkan::Allocation
}