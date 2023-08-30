use daxa_rs::{
    context::*,
    device::*,
    gpu_resources::*,
    memory_block::AllocationInfo,
    semaphore::*,
    split_barrier::*,
    util::mem::*, command_list::CommandListInfo,
};

use std::path::PathBuf;

const APPNAME: &str = "Daxa API Test: Mem";
const APPNAME_PREFIX: &str = "[Daxa API Test: Mem]";

const ITERATION_COUNT: usize = 1000;
const ELEMENT_COUNT: usize = 17;

#[test]
fn main() {
    let context = Context::new(ContextInfo {
        application_name: format!("{} (main)", APPNAME).into(),
        application_version: 1,
        ..Default::default()
    }).unwrap();

    let device = context.create_device(DeviceInfo {
        debug_name: format!("{} device (main)", APPNAME_PREFIX).into(),
        ..Default::default()
    }).unwrap();

    let mut tmem = TransferMemoryPool::new(device.clone(), TransferMemoryPoolInfo {
        capacity: 256,
        debug_name: "transient memory pool".into()
    }).unwrap();

    let gpu_timeline = device.create_timeline_semaphore(TimelineSemaphoreInfo {
        debug_name: "timeline semaphore".into(),
        ..Default::default()
    }).unwrap();

    let cpu_timeline = 1;

    let result_buffer = device.create_buffer(BufferInfo {
        size: (std::mem::size_of::<u32>() * ELEMENT_COUNT * ITERATION_COUNT) as u32,
        allocation_info: AllocationInfo::Automatic(MemoryLocation::CpuToGpu),
        debug_name: "result".into()
    }).unwrap();

    for i in 0..ITERATION_COUNT {
        gpu_timeline.wait_for_value(cpu_timeline - 1, 0);

        let mut cmd = device.create_command_list(CommandListInfo { 
            debug_name: format!("{} (main {})", APPNAME_PREFIX, i).into()
        }).unwrap();

        cmd.pipeline_barrier(MemoryBarrierInfo {
            src_access: daxa_rs::types::access_consts::TRANSFER_READ_WRITE | daxa_rs::types::access_consts::HOST_WRITE,
            dst_access: daxa_rs::types::access_consts::TRANSFER_READ_WRITE
        });

        // Can allocate anywhere in the frame with immediately available staging memory
        let alloc = tmem.allocate(ELEMENT_COUNT as u32, 8).unwrap();
        for j in 0..ELEMENT_COUNT {
            // The allocation provides a host pointer to the memory
            //alloc.mapped_slice()
        }
    }
}