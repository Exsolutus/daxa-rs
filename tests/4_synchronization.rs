use daxa_rs::{context::*, device::*, command_list::*};
use daxa_rs::semaphore::BinarySemaphoreInfo;

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
    let command_list1 = command_list1.complete().unwrap();

    let command_list2 = app.device.create_command_list(CommandListInfo::default()).unwrap();
    let command_list2 = command_list2.complete().unwrap();

    let command_list3 = app.device.create_command_list(CommandListInfo::default()).unwrap();
    let command_list3 = command_list3.complete().unwrap();

    let command_list4 = app.device.create_command_list(CommandListInfo::default()).unwrap();
    let command_list4 = command_list4.complete().unwrap();

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