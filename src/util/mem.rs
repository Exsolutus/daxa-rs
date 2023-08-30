use crate::{
    core::*,
    device::Device,
    gpu_resources::{
        BufferId,
        BufferInfo
    },
    memory_block::AllocationInfo,
    semaphore::{
        TimelineSemaphore,
        TimelineSemaphoreInfo
    }, 
};

use anyhow::Result;
use ash::vk::{
    DeviceAddress
};
use gpu_allocator::MemoryLocation;
use std::{
    borrow::Cow,
    collections::VecDeque,
};


#[derive(Clone)]
pub struct TransferMemoryPoolInfo {
    pub capacity: u32,
    //use_bar_memory: bool,
    pub debug_name:  Cow<'static, str>
}

impl Default for TransferMemoryPoolInfo {
    fn default() -> Self {
        Self {
            capacity: 1 << 25,
            //use_bar_memory: false,
            debug_name: "".into()
        }
    }
}

struct Suballocation {
    timeline_index: usize,
    offset: u32,
    size: u32
}

pub struct Allocation {
    //allocation: gpu_allocator::vulkan::Allocation,
    buffer_device_address: DeviceAddress,
    //buffer_host_address: std::ptr::NonNull<std::ffi::c_void>,
    offset: u32,
    size: usize,
    timeline_index: u64
}

// impl std::ops::Deref for Allocation {
//     type Target = gpu_allocator::vulkan::Allocation;

//     fn deref(&self) -> &Self::Target {
//         &self.allocation
//     }
// }


/// Ring buffer based transfer memory allocator for easy and efficient cpu-gpu communication.
pub struct TransferMemoryPool {
    device: Device,
    info: TransferMemoryPoolInfo,
    gpu_timeline: TimelineSemaphore,

    current_timeline_value: u64,
    live_allocations: VecDeque<Suballocation>,
    buffer: BufferId,
    buffer_device_address: DeviceAddress,
    buffer_host_address: std::ptr::NonNull<std::ffi::c_void>,
    claimed_start: u32,
    claimed_size: u32
}

// TransferMemoryPool creation methods
impl TransferMemoryPool {
    pub fn new(device: Device, info: TransferMemoryPoolInfo) -> Result<Self> {
        let gpu_timeline = device.create_timeline_semaphore(TimelineSemaphoreInfo {
            debug_name: format!("TransferMemoryPool {}", info.debug_name).into(),
            ..Default::default()
        })?;

        let buffer = device.create_buffer(BufferInfo {
            size: info.capacity,
            allocation_info: AllocationInfo::Automatic(MemoryLocation::CpuToGpu),
            debug_name: format!("TransferMemoryPool {}", info.debug_name).into()
        })?;

        let buffer_device_address = device.get_device_address(buffer);

        let buffer_host_address = device.get_host_address(buffer).unwrap();

        Ok(Self {
            device,
            info,
            gpu_timeline,
            current_timeline_value: Default::default(),
            live_allocations: Default::default(),
            buffer,
            buffer_device_address,
            buffer_host_address,
            claimed_start: 0,
            claimed_size: 0
        })
    }
}

// TransferMemoryPool usage methods
impl TransferMemoryPool {
    pub fn allocate(&mut self, size: u32, alignment: u32) -> Option<Allocation> {
        let tail_alloc_offset = (self.claimed_start + self.claimed_size) % self.info.capacity;
        let tail_alloc_offset_aligned = (tail_alloc_offset + alignment - 1) / alignment * alignment;
        let tail_alloc_align_padding = tail_alloc_offset_aligned - tail_alloc_offset;
        // Two allocations are possible:
        // Tail allocation is when the allocation is placed directly at the end of all other allocations.
        // Zero offset allocation is possible when there is not enough space left at the tail BUT there is enough space from 0 up to the start of the other allocations.
        let calc_tail_alloc_possible = |start: u32, size: u32, capacity: u32| {
            let wrapped = start + size > capacity;
            let end = match wrapped {
                true => start,
                false => capacity
            };
            tail_alloc_offset_aligned + size <= end
        };
        let calc_zero_offset_alloc_possible = |start: u32, size: u32, capacity: u32| {
            start + size <= capacity && size < start
        };
        // Firstly, test if there is enough continuous space left to allocate.
        let mut tail_alloc_possible = calc_tail_alloc_possible(self.claimed_start, self.claimed_size, self.info.capacity);
        // When there is no tail space left, it may be the case that we can place the allocation at offset 0.
        // Illustration: |XXX ## |; "X": new allocation; "#": used up space; " ": free space.
        let mut zero_offset_alloc_possible = calc_zero_offset_alloc_possible(self.claimed_start, self.claimed_size, self.info.capacity);
        if !tail_alloc_possible && !zero_offset_alloc_possible {
            self.reclaim_unused_memory();
            tail_alloc_possible = calc_tail_alloc_possible(self.claimed_start, self.claimed_size, self.info.capacity);
            zero_offset_alloc_possible = calc_zero_offset_alloc_possible(self.claimed_start, self.claimed_size, self.info.capacity);
            if !tail_alloc_possible && !zero_offset_alloc_possible {
                return None;
            }
        }
        self.current_timeline_value += 1;
        let mut returned_alloc_offset = 0;
        let mut actual_alloc_offset = 0;
        let mut actual_alloc_size = 0;
        if tail_alloc_possible {
            actual_alloc_size = size + tail_alloc_align_padding;
            returned_alloc_offset = tail_alloc_offset_aligned;
            actual_alloc_offset = tail_alloc_offset;
        } else {
            let left_tail_space = self.info.capacity - (self.claimed_start + self.claimed_size);
            actual_alloc_size = size + left_tail_space;
        }
        self.claimed_size += actual_alloc_size;
        self.live_allocations.push_back(Suballocation {
            timeline_index: self.current_timeline_value as usize,
            offset: actual_alloc_offset,
            size: actual_alloc_size
        });

        Some(Allocation {
            buffer_device_address: self.buffer_device_address + returned_alloc_offset as u64,
            //buffer_host_address: unsafe { self.buffer_host_address.as_ptr().offset(returned_alloc_offset). },
            offset: returned_alloc_offset,
            size: size as usize,
            timeline_index: self.current_timeline_value 
        })
    }

    pub fn timeline_value(&self) -> usize {
        self.current_timeline_value as usize
    }

    pub fn get_timeline_semaphore(&self) -> TimelineSemaphore {
        self.gpu_timeline.clone()
    }
    
    pub fn get_info(&self) -> TransferMemoryPoolInfo {
        self.info.clone()
    }

    pub fn get_buffer(&self) -> BufferId {
        self.buffer
    }
}

// TransferMemoryPool internal methods
impl TransferMemoryPool {
    fn reclaim_unused_memory(&mut self) {
        let current_gpu_timeline_value = self.gpu_timeline.value() as usize;

        while !self.live_allocations.is_empty() && self.live_allocations.front().unwrap().timeline_index <= current_gpu_timeline_value {
            let front = self.live_allocations.front().unwrap();
            self.claimed_start = (self.claimed_start + front.size) % self.info.capacity;
            self.claimed_size -= front.size;
            self.live_allocations.pop_front();
        }
    }
}

impl Drop for TransferMemoryPool {
    fn drop(&mut self) {
        self.device.destroy_buffer(self.buffer);
    }
}