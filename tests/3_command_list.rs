use daxa_rs::{types::*, context::*, device::*, gpu_resources::*, timeline_query::*, split_barrier::*};
use daxa_rs::command_list::*;
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::mem::size_of;



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
fn simplest() {
    let app = App::new();

    let command_list = app.device.create_command_list(CommandListInfo::default()).unwrap();

    // Command lists must be completed before submission!
    let command_list = command_list.complete().unwrap();

    app.device.submit_commands(CommandSubmitInfo {
        command_lists: vec![command_list],
        ..Default::default()
    })
}

#[test]
fn deferred_destruction() {
    let app = App::new();

    let mut command_list = app.device.create_command_list(CommandListInfo {
        debug_name: "deferred_destruction command list".into()
    }).unwrap();

    let buffer = app.device.create_buffer(BufferInfo {
        size: 4,
        ..Default::default()
    }).unwrap();

    let image = app.device.create_image(ImageInfo {
        size: vk::Extent3D { width: 1, height: 1, depth: 1 },
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ..Default::default()
    }).unwrap();

    let image_view = app.device.create_image_view(ImageViewInfo {
        image,
        ..Default::default()
    }).unwrap();

    let sampler = app.device.create_sampler(SamplerInfo {
        ..Default::default()
    }).unwrap();

    // The gpu resources are not destroyed here. Their destruction is deferred until the command list completes execution on the gpu.
    command_list.destroy_buffer_deferred(buffer);
    command_list.destroy_image_deferred(image);
    command_list.destroy_image_view_deferred(image_view);
    command_list.destroy_sampler_deferred(sampler);

    // The gpu resources are still alive, as long as this command list is not submitted and has not finished execution.
    let command_list = command_list.complete().unwrap();

    // Even after this call the resources will still be alive, as zombie resources are not checked to be dead in submit calls.
    app.device.submit_commands(CommandSubmitInfo {
        command_lists: vec![command_list],
        ..Default::default()
    });

    app.device.wait_idle();

    // Here the gpu resources will be destroyed.
    // Collect_garbage loops over all zombie resources and destroys them when they are no longer used on the gpu/ their associated command list finished executing.
    app.device.collect_garbage();
}

#[test]
fn copy() {
    let app = App::new();

    let mut command_list = app.device.create_command_list(CommandListInfo {
        debug_name: "copy command list".into()
    }).unwrap();

    const SIZE_X: u32 = 3;
    const SIZE_Y: u32 = 3;
    const SIZE_Z: u32 = 3;

    type ImageArray = [[[[f32; 4]; SIZE_X as usize]; SIZE_Y as usize]; SIZE_Z as usize];

    let mut data = ImageArray::default();

    for zi in 0..SIZE_Z {
        for yi in 0..SIZE_Y {
            for xi in 0..SIZE_X {
                data[zi as usize][yi as usize][xi as usize] = [
                    (xi as f32) / ((SIZE_X - 1) as f32),
                    (yi as f32) / ((SIZE_Y - 1) as f32),
                    (zi as f32) / ((SIZE_Z - 1) as f32),
                    1.0
                ]
            }
        }
    }

    let staging_upload_buffer = app.device.create_buffer(BufferInfo {
        memory_location: MemoryLocation::CpuToGpu,
        size: size_of::<ImageArray>() as u32,
        debug_name: "staging_upload_buffer".into(),
    }).unwrap();

    let device_local_buffer = app.device.create_buffer(BufferInfo {
        memory_location: MemoryLocation::GpuOnly,
        size: size_of::<ImageArray>() as u32,
        debug_name: "device_local_buffer".into(),
    }).unwrap();

    let staging_readback_buffer = app.device.create_buffer(BufferInfo {
        memory_location: MemoryLocation::GpuToCpu,
        size: size_of::<ImageArray>() as u32,
        debug_name: "staging_readback_buffer".into(),
    }).unwrap();

    let image_1 = app.device.create_image(ImageInfo {
        dimensions: match SIZE_Z > 1 { true => 3, false => 2 },
        format: vk::Format::R32G32B32A32_SFLOAT,
        size: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
        usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        debug_name: "image_1".into(),
        ..Default::default()
    }).unwrap();

    let image_2 = app.device.create_image(ImageInfo {
        dimensions: match SIZE_Z > 1 { true => 3, false => 2 },
        format: vk::Format::R32G32B32A32_SFLOAT,
        size: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
        usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        debug_name: "image_2".into(),
        ..Default::default()
    }).unwrap();

    let timeline_query_pool = app.device.create_timeline_query_pool(TimelineQueryPoolInfo {
        query_count: 2,
        debug_name: "timeline_query".into()
    }).unwrap();

    let buffer_ptr = unsafe {
        app.device.get_host_address_as::<ImageArray>(staging_upload_buffer)
            .unwrap()
            .as_mut()
    };

    *buffer_ptr = data;

    command_list.reset_timestamps(ResetTimestampsInfo {
        query_pool: timeline_query_pool.clone(),
        start_index: 0,
        count: timeline_query_pool.info().query_count
    });

    command_list.write_timestamp(WriteTimestampInfo {
        query_pool: timeline_query_pool.clone(),
        stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        query_index: 0
    });

    command_list.pipeline_barrier(MemoryBarrierInfo {
        awaited_pipeline_access: access_consts::HOST_WRITE,
        waiting_pipeline_access: access_consts::TRANSFER_READ
    });

    command_list.copy_buffer_to_buffer(BufferCopyInfo {
        src_buffer: staging_upload_buffer,
        dst_buffer: device_local_buffer,
        size: size_of::<ImageArray>() as vk::DeviceSize,
        ..Default::default()
    });

    // Barrier to make sure device_local_buffer is has no read after write hazard.
    command_list.pipeline_barrier(MemoryBarrierInfo {
        awaited_pipeline_access: access_consts::TRANSFER_WRITE,
        waiting_pipeline_access: access_consts::TRANSFER_READ
    });

    command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
        awaited_pipeline_access: access_consts::TRANSFER_WRITE,
        waiting_pipeline_access: access_consts::TRANSFER_WRITE,
        after_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image: image_1,
        ..Default::default()
    });

    command_list.copy_buffer_to_image(BufferImageCopyInfo {
        buffer: device_local_buffer,
        image: image_1,
        image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image_extent: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
        ..Default::default()
    });

    command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
        awaited_pipeline_access: access_consts::TRANSFER_WRITE,
        waiting_pipeline_access: access_consts::TRANSFER_READ,
        after_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        image: image_1,
        ..Default::default()
    });

    command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
        waiting_pipeline_access: access_consts::TRANSFER_WRITE,
        after_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image: image_2,
        ..Default::default()
    });

    command_list.copy_image_to_image(ImageCopyInfo {
        src_image: image_1,
        src_image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        dst_image: image_2,
        dst_image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        extent: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
        ..Default::default()
    });

    command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
        awaited_pipeline_access: access_consts::TRANSFER_WRITE,
        waiting_pipeline_access: access_consts::TRANSFER_READ,
        after_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        image: image_2,
        ..Default::default()
    });

    // Barrier to make sure device_local_buffer is has no write after read hazard.
    command_list.pipeline_barrier(MemoryBarrierInfo {
        awaited_pipeline_access: access_consts::TRANSFER_READ,
        waiting_pipeline_access: access_consts::TRANSFER_WRITE
    });

    command_list.copy_image_to_buffer(ImageBufferCopyInfo {
        image: image_2,
        image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        image_extent: vk::Extent3D { width: SIZE_X, height: SIZE_Y, depth: SIZE_Z },
        buffer: device_local_buffer,
        ..Default::default()
    });

    // Barrier to make sure device_local_buffer is has no read after write hazard.
    command_list.pipeline_barrier(MemoryBarrierInfo {
        awaited_pipeline_access: access_consts::TRANSFER_WRITE,
        waiting_pipeline_access: access_consts::TRANSFER_READ
    });

    command_list.copy_buffer_to_buffer(BufferCopyInfo {
        src_buffer: device_local_buffer,
        dst_buffer: staging_readback_buffer,
        size: size_of::<ImageArray>() as vk::DeviceSize,
        ..Default::default()
    });

    // Barrier to make sure staging_readback_buffer is has no read after write hazard.
    command_list.pipeline_barrier(MemoryBarrierInfo {
        awaited_pipeline_access: access_consts::TRANSFER_WRITE,
        waiting_pipeline_access: access_consts::HOST_READ
    });

    command_list.write_timestamp(WriteTimestampInfo {
        query_pool: timeline_query_pool.clone(),
        stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        query_index: 1
    });

    let command_list = command_list.complete().unwrap();

    app.device.submit_commands(CommandSubmitInfo {
        command_lists: vec![command_list],
        ..Default::default()
    });

    app.device.wait_idle();

    // Validate and display results

    let query_results = timeline_query_pool.get_query_results(0, 2).unwrap();
    if query_results[0].1 != 0 && query_results[1].1 != 0 {
        println!("GPU execution took {} ms", ((query_results[1].0 - query_results[0].0) as f64) / 1000000.0);
    }

    let readback_data = unsafe {
        app.device.get_host_address_as::<ImageArray>(staging_upload_buffer)
            .unwrap()
            .as_ref()
    };

    fn get_printable_char_buffer(in_data: &ImageArray) -> String {
        const PIXEL: &str = "\x1B[48;2;000;000;000m  ";
        const LINE_TERMINATOR: &str = "\x1B[0m ";
        const NEWLINE_TERMINATOR: &str = "\x1B[0m\n";

        let capacity: usize = (SIZE_X * SIZE_Y * SIZE_Z) as usize * (PIXEL.len() - 1)
                            + (SIZE_Y * SIZE_Z) as usize * (LINE_TERMINATOR.len() - 1)
                            + SIZE_Z as usize * (NEWLINE_TERMINATOR.len() - 1)
                            + 1;
        let mut data = String::with_capacity(capacity);

        for zi in 0..SIZE_Z as usize {
            for yi in 0..SIZE_Y as usize {
                for xi in 0..SIZE_X as usize {
                    let r = (in_data[zi][yi][xi][0] * 255.0) as u8;
                    let g = (in_data[zi][yi][xi][1] * 255.0) as u8;
                    let b = (in_data[zi][yi][xi][2] * 255.0) as u8;
                    let mut next_pixel = String::from(PIXEL).into_bytes();
                    next_pixel[7 + 0 * 4 + 0] = 48 + (r / 100);
                    next_pixel[7 + 0 * 4 + 1] = 48 + ((r % 100) / 10);
                    next_pixel[7 + 0 * 4 + 2] = 48 + (r % 10);
                    next_pixel[7 + 1 * 4 + 0] = 48 + (g / 100);
                    next_pixel[7 + 1 * 4 + 1] = 48 + ((g % 100) / 10);
                    next_pixel[7 + 1 * 4 + 2] = 48 + (g % 10);
                    next_pixel[7 + 2 * 4 + 0] = 48 + (b / 100);
                    next_pixel[7 + 2 * 4 + 1] = 48 + ((b % 100) / 10);
                    next_pixel[7 + 2 * 4 + 2] = 48 + (b % 10);
                    let next_pixel = String::from_utf8(next_pixel).unwrap();
                    data.push_str(&next_pixel);
                }
                data.push_str(LINE_TERMINATOR);
            }
            data.push_str(NEWLINE_TERMINATOR);
        }
        
        data.to_ascii_lowercase()
    }

    println!("Original data:\n{}", get_printable_char_buffer(&data));
    println!("Readback data:\n{}", get_printable_char_buffer(&readback_data));

    #[cfg(debug_assertions)]
    for zi in 0..SIZE_Z {
        for yi in 0..SIZE_Y {
            for xi in 0..SIZE_X {
                for ci in 0..4 {
                    debug_assert_eq!(
                        data[zi as usize][yi as usize][xi as usize][ci as usize],
                        readback_data[zi as usize][yi as usize][xi as usize][ci as usize],
                        "Readback data differs from upload data."
                    )
                }
            }
        }
    }

    app.device.destroy_buffer(staging_upload_buffer);
    app.device.destroy_buffer(device_local_buffer);
    app.device.destroy_buffer(staging_readback_buffer);
    app.device.destroy_image(image_1);
    app.device.destroy_image(image_2);

    app.device.collect_garbage();
}