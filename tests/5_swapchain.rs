use daxa_rs::{
    core::*,
    types::*,
    context::*,
    device::*,
    command_list::*,
    split_barrier::*,
    swapchain::*, 
};

mod common;
use common::window::AppWindow;

use ash::vk;

use std::{
    slice
};


const APPNAME: &str = "Daxa API Test: Swapchain";
const APPNAME_PREFIX: &str = "[Daxa API Test: Swapchain]";



#[test]
fn simple_creation() {
    struct App {
        window: AppWindow,
        context: Context,
        device: Device,
        swapchain: Swapchain
    }

    impl App {
        pub fn new() -> Self {
            let window  = AppWindow::new(format!("{} (simple_creation)", APPNAME).as_str());

            let context = Context::new(ContextInfo {
                application_name: format!("{} context (simple_creation)", APPNAME_PREFIX).into(),
                application_version: 1,
                ..Default::default()
            }).unwrap();

            let device = context.create_device(DeviceInfo {
                debug_name: format!("{} device (simple_creation)", APPNAME_PREFIX).into(),
                ..Default::default()
            }).unwrap();

            let swapchain = device.create_swapchain(SwapchainInfo {
                raw_window_handle: window.get_raw_window(),
                raw_display_handle: window.get_raw_display(),
                present_mode: vk::PresentModeKHR::FIFO,
                image_usage: vk::ImageUsageFlags::TRANSFER_DST,
                debug_name: format!("{} swapchain (simple_creation)", APPNAME_PREFIX).into(),
                ..Default::default()
            }).unwrap();

            Self {
                window,
                context,
                device,
                swapchain
            }
        }
    }

    App::new();
}

#[test]
fn clear_color() {
    struct App {
        window: AppWindow,
        context: Context,
        device: Device,
        swapchain: Swapchain
    }

    impl App {
        pub fn new() -> Self {
            let window  = AppWindow::new(format!("{} (clear_color)", APPNAME).as_str());

            let context = Context::new(ContextInfo {
                application_name: format!("{} context (clear_color)", APPNAME_PREFIX).into(),
                application_version: 1,
                ..Default::default()
            }).unwrap();

            let device = context.create_device(DeviceInfo {
                debug_name: format!("{} device (clear_color)", APPNAME_PREFIX).into(),
                ..Default::default()
            }).unwrap();

            let swapchain = device.create_swapchain(SwapchainInfo {
                raw_window_handle: window.get_raw_window(),
                raw_display_handle: window.get_raw_display(),
                present_mode: vk::PresentModeKHR::FIFO,
                image_usage: vk::ImageUsageFlags::TRANSFER_DST,
                debug_name: format!("{} swapchain (clear_color)", APPNAME_PREFIX).into(),
                ..Default::default()
            }).unwrap();

            Self {
                window,
                context,
                device,
                swapchain
            }
        }

        pub fn update(&mut self) -> bool {
            let glfw = &mut self.window.glfw;
            glfw.poll_events();

            if self.window.should_close() {
                return true;
            }

            for (_, event) in glfw::flush_messages(&self.window.events) {
                match event {
                    glfw::WindowEvent::FramebufferSize(_width, _height) => {
                        self.on_resize()
                    },
                    _ => ()
                }
            }

            if !self.window.is_iconified() {
                self.draw();
            }
            else {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }

            false
        }

        pub fn draw(&self) {
            let device = &self.device;
            let swapchain = &self.swapchain;

            let swapchain_image = swapchain.acquire_next_image();
            if swapchain_image.is_empty() {
                return;
            }

            let mut command_list = device.create_command_list(CommandListInfo {
                debug_name: format!("{} command_list (clear_color)", APPNAME_PREFIX).into(),
            }).unwrap();

            command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
                waiting_pipeline_access: access_consts::TRANSFER_WRITE,
                before_layout: vk::ImageLayout::UNDEFINED,
                after_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: swapchain_image,
                ..Default::default()
            });

            command_list.clear_image(&ImageClearInfo {
                image: swapchain_image,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                clear_value: vk::ClearValue { color: vk::ClearColorValue { float32: [1.0, 0.0, 1.0, 1.0] } },
                ..Default::default()
            });

            command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
                waiting_pipeline_access: access_consts::TRANSFER_WRITE,
                before_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                after_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                image: swapchain_image,
                ..Default::default()
            });

            let command_list = command_list.complete().unwrap();

            device.submit_commands(CommandSubmitInfo {
                command_lists: vec![command_list],
                wait_binary_semaphores: vec![swapchain.get_acquire_semaphore()],
                signal_binary_semaphores: vec![swapchain.get_present_semaphore()],
                signal_timeline_semaphores: vec![(swapchain.get_gpu_timeline_semaphore(), swapchain.get_cpu_timeline_value() as u64)],
                ..Default::default()
            });

            device.preset_frame(PresentInfo {
                wait_binary_semaphores: vec![swapchain.get_present_semaphore()],
                swapchain
            });
        }

        pub fn on_resize(&self) {
            if !self.window.is_iconified() {
                self.swapchain.resize();
                self.draw();
            }
        }
    }

    let mut app = App::new();
    loop {
        if app.update() {
            break;
        }
    }
}