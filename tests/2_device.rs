
use daxa_rs::context::{Context, ContextInfo};
use daxa_rs::device::{DeviceInfo, DeviceType};
use std::ffi::CStr;

fn context() -> Context {
    Context::new(ContextInfo::default()).unwrap()
}

#[test]
fn simplest() {
    let daxa_context = context();

    let device = daxa_context.create_device(DeviceInfo::default());

    assert!(device.is_ok())
}

#[test]
fn device_selection() {
    let daxa_context = context();

    // To select a device, you look at its properties and return a score.
    // Daxa will choose the device you scored as the highest.
    let device = daxa_context.create_device(DeviceInfo {
        selector: |&properties| {
            let mut score = 0;

            match properties.device_type {
                DeviceType::DISCRETE_GPU => score += 10000,
                DeviceType::VIRTUAL_GPU => score += 1000,
                DeviceType::INTEGRATED_GPU => score += 100,
                _ => ()
            }

            score
        },
        debug_name: "My device",
    });

    assert!(device.is_ok());

    // Once the device is created, you can query its properties, such
    // as its name and much more! These are the same properties we used
    // to discriminate in the GPU selection.
    unsafe { println!("{:?}", CStr::from_ptr(device.unwrap().properties().device_name.as_ptr())) }
}
