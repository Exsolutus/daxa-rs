use glfw::*;
use raw_window_handle::{
    RawDisplayHandle,
    RawWindowHandle,
    HasRawWindowHandle,
    HasRawDisplayHandle
};

use std::{
    sync::mpsc::Receiver,
    ops::{Deref, DerefMut}
};


pub struct AppWindow {
    window: Window,
    pub events: Receiver<(f64, WindowEvent)>,
}

impl Deref for AppWindow {
    type Target = Window;

    fn deref(&self) -> &Self::Target {
        &self.window
    }
}

impl DerefMut for AppWindow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.window
    }
}

impl AppWindow {
    pub fn new(window_name: &str) -> Self {
        AppWindow::with_size(window_name, 800, 600)
    }

    pub fn with_size(window_name: &str, width: u32, height: u32) -> Self {
        let mut glfw = glfw::init(FAIL_ON_ERRORS).unwrap();
        glfw.window_hint(WindowHint::Visible(true));
        glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));

        let (mut window, events) = glfw.create_window(
            width,
            height, 
            window_name,
            WindowMode::Windowed
        ).expect("Failed to create GLFW window.");

        window.set_cursor_pos_polling(true);
        window.set_mouse_button_polling(true);
        window.set_key_polling(true);
        window.set_framebuffer_size_polling(true);

        Self {
            window,
            events,
        }
    }

    pub fn get_raw_window(&self) -> RawWindowHandle {
        self.window.raw_window_handle()
    }

    pub fn get_raw_display(&self) -> RawDisplayHandle {
        self.window.raw_display_handle()
    }

    pub fn set_mouse_pos(&mut self, x: f32, y: f32) {
        self.window.set_cursor_pos(x as f64, y as f64);
    }

    pub fn set_mouse_capture(&mut self, should_capture: bool) {
        let (width, height) = self.window.get_size();
        self.window.set_cursor_pos((width / 2) as f64, (height / 2) as f64);
        self.window.set_cursor_mode(match should_capture {
            true => CursorMode::Normal,
            false => CursorMode::Disabled
        });
        self.window.set_raw_mouse_motion(should_capture);
    }
}
