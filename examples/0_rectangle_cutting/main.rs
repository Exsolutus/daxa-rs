use common::window;

use rectangle_cutting_shaders_shared::*;

use daxa_rs::{
    core::*,
    util::pipeline_manager
};

use ash::vk; // TODO: reexport whatever we use from here through daxa_rs
use spirv_std::glam;

use std::cell::Cell;



const APPNAME: &str = "Daxa Sample: Rectangle Cutting";
const APPNAME_PREFIX: &str = "[Daxa Sample: Rectangle Cutting]";

const MAX_LAYERS: i32 = 12;
const MAX_LEVELS: i32 = 16;

const MAX_VERTS: u32 = 10000;



fn main() {
    let mut app = App::new();
    loop {
        if app.update() {
            break;
        }
    }
}



struct App {
    window: window::AppWindow,
    context: daxa_rs::context::Context,
    device: daxa_rs::device::Device,
    swapchain: daxa_rs::swapchain::Swapchain,
    pipeline_manager: pipeline_manager::PipelineManager,
    raster_pipeline: daxa_rs::pipeline::RasterPipeline,
    vertex_buffer: daxa_rs::gpu_resources::BufferId,
    vertex_count: Cell<u32>,
    range_0: daxa_rs::gpu_resources::ImageSubresourceRange,
    range_1: daxa_rs::gpu_resources::ImageSubresourceRange,
}

impl App {
    pub fn new() -> Self {
        let window = window::AppWindow::new(format!("{}", APPNAME).as_str());

        let context = daxa_rs::context::Context::new(daxa_rs::context::ContextInfo {
            application_name: format!("{}", APPNAME).into(),
            application_version: 1,
            ..Default::default()
        }).unwrap();

        let device = context.create_device(daxa_rs::device::DeviceInfo {
            debug_name: format!("{} device", APPNAME_PREFIX).into(),
            ..Default::default()
        }).unwrap();

        let swapchain = device.create_swapchain(daxa_rs::swapchain::SwapchainInfo {
            raw_window_handle: window.get_raw_window(),
            raw_display_handle: window.get_raw_display(),
            present_mode: vk::PresentModeKHR::IMMEDIATE,
            image_usage: vk::ImageUsageFlags::TRANSFER_DST,
            debug_name: format!("{} swapchain", APPNAME_PREFIX).into(),
            ..Default::default()
        }).unwrap();

        use pipeline_manager::PipelineManager;
        use pipeline_manager::PipelineManagerInfo;
        use pipeline_manager::ShaderCompileOptions;
        use std::path::PathBuf;

        let mut pipeline_manager = PipelineManager::new(PipelineManagerInfo {
            device: device.clone(),
            shader_compile_options: ShaderCompileOptions {
                root_paths: vec![
                    ["examples", "0_rectangle_cutting"].iter().collect::<PathBuf>(),
                ],
                ..Default::default()
            },
            debug_name: format!("{} pipeline_manager", APPNAME_PREFIX).into() 
        }).unwrap();

        use pipeline_manager::RasterPipelineCompileInfo;
        use pipeline_manager::ShaderCompileInfo;
        use pipeline_manager::ShaderCrate;
        use daxa_rs::pipeline;
        use std::mem::size_of;

        let raster_pipeline = pipeline_manager.add_raster_pipeline(RasterPipelineCompileInfo {
            vertex_shader_info: ShaderCompileInfo {
                source: ShaderCrate("shaders".into()).into(),
                compile_options: ShaderCompileOptions { entry_point: "main_vs", ..Default::default() }
            },
            fragment_shader_info: ShaderCompileInfo {
                source: ShaderCrate("shaders".into()).into(),
                compile_options: ShaderCompileOptions { entry_point: "main_fs", ..Default::default() }
            },
            color_attachments: vec![
                pipeline::RenderAttachment {
                    format: swapchain.get_format(),
                    blend: pipeline::PipelineColorBlendAttachmentState {
                        blend_enable: 1,
                        src_color_blend_factor: pipeline::BlendFactor::SRC_ALPHA,
                        dst_color_blend_factor: pipeline::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        src_alpha_blend_factor: pipeline::BlendFactor::ONE,
                        dst_alpha_blend_factor: pipeline::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        ..Default::default()
                    }
                }
            ],
            depth_test: pipeline::DepthTestInfo::default(),
            raster: pipeline::RasterizerInfo::default(),
            push_constant_index: (1) as u32, // TODO: handle this better
            debug_name: format!("{} raster_pipeline", APPNAME_PREFIX).into()
        }).unwrap();

        use daxa_rs::gpu_resources::BufferInfo;
        use daxa_rs::gpu_resources::MemoryLocation;

        let vertex_buffer = device.create_buffer(BufferInfo {
            memory_location: MemoryLocation::GpuOnly,
            size: MAX_VERTS * size_of::<DrawVertex>() as u32,
            debug_name: format!("{} vertex_buffer", APPNAME_PREFIX).into(),
        }).unwrap();

        use daxa_rs::gpu_resources::ImageSubresourceRange;
        use daxa_rs::gpu_resources::ImageAspectFlags;

        let range_0 = ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR | ImageAspectFlags::DEPTH,
            base_mip_level: 3,
            level_count: 5,
            base_array_layer: 2,
            layer_count: 4
        };

        let range_1 = ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 3,
            level_count: 5,
            base_array_layer: 2,
            layer_count: 4
        };

        Self {
            window,
            context,
            device,
            swapchain,
            pipeline_manager,
            raster_pipeline,
            vertex_buffer,
            vertex_count: 0.into(),
            range_0,
            range_1
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

    fn add_rectangle(&self, buffer_slice: &mut [DrawVertex], p0: glam::Vec2, p1: glam::Vec2, color: glam::Vec4) {
        let n = self.vertex_count.get();
        buffer_slice[n as usize] = DrawVertex { position: glam::vec4(p0.x, p0.y, 0.0, 0.0), color };
        buffer_slice[(n + 1) as usize] = DrawVertex { position: glam::vec4(p1.x, p0.y, 0.0, 0.0), color };
        buffer_slice[(n + 2) as usize] = DrawVertex { position: glam::vec4(p0.x, p1.y, 0.0, 0.0), color };

        buffer_slice[(n + 3) as usize] = DrawVertex { position: glam::vec4(p1.x, p0.y, 0.0, 0.0), color };
        buffer_slice[(n + 4) as usize] = DrawVertex { position: glam::vec4(p0.x, p1.y, 0.0, 0.0), color };
        buffer_slice[(n + 5) as usize] = DrawVertex { position: glam::vec4(p1.x, p1.y, 0.0, 0.0), color };

        self.vertex_count.set(n + 6);
    }

    pub fn ui_update(&self) {
        todo!()
    }

    pub fn construct_scene(&self, buffer_slice: &mut [DrawVertex]) {
        self.vertex_count.set(0);

        use daxa_rs::types::{
            Rect2D,
            Offset2D,
            Extent2D
        };

        fn view_transform(v: glam::Vec2) -> glam::Vec2 {
            (v / glam::vec2(MAX_LEVELS as f32, MAX_LAYERS as f32)) * 2.0 - 1.0
        }
        let mut add_int_rectangle = |rectange: Rect2D, scale: f32, color: glam::Vec4| {
            let p0 = glam::vec2(rectange.offset.x as f32, rectange.offset.y as f32) + (scale * 0.5);
            let p1 = p0 + glam::vec2(rectange.extent.width as f32, rectange.extent.height as f32) - scale;
            self.add_rectangle(buffer_slice, view_transform(p0), view_transform(p1), color)
        };

        for y in 0..MAX_LAYERS {
            for x in 0..MAX_LEVELS {
                add_int_rectangle(
                    Rect2D { offset: Offset2D { x, y }, extent: Extent2D { width: 1, height: 1 } },
                    0.1,
                    glam::vec4(0.1, 0.1, 0.1, 0.5)
                )
            }
        }

        add_int_rectangle(
            Rect2D { 
                offset: Offset2D { x: self.range_0.base_mip_level as i32, y: self.range_0.base_array_layer as i32 }, 
                extent: Extent2D { width: self.range_0.level_count, height: self.range_0.layer_count } },
            0.0,
            glam::vec4(0.3, 0.9, 0.3, 0.9)
        );
        add_int_rectangle(
            Rect2D { 
                offset: Offset2D { x: self.range_1.base_mip_level as i32, y: self.range_1.base_array_layer as i32 }, 
                extent: Extent2D { width: self.range_1.level_count, height: self.range_1.layer_count } },
            0.0,
            glam::vec4(0.9, 0.3, 0.3, 0.9)
        );

        let (range2_rectangles, range2_rectangles_n) = self.range_0.subtract(self.range_1);
        let range2_colors = [
            glam::vec4(0.1, 0.1, 0.1, 0.5),
            glam::vec4(0.1, 0.1, 0.1, 0.5),
            glam::vec4(0.1, 0.1, 0.1, 0.5),
            glam::vec4(0.1, 0.1, 0.1, 0.5),
        ];

        for i in 0..range2_rectangles_n {
            let range_2 = range2_rectangles[i];
            add_int_rectangle(
                Rect2D { 
                    offset: Offset2D { x: range_2.base_mip_level as i32, y: range_2.base_array_layer as i32 }, 
                    extent: Extent2D { width: range_2.level_count, height: range_2.layer_count } },
                0.2,
                range2_colors[i]
            );
        }
    }

    pub fn draw(&self) {
        //self.ui_update();

        let device = &self.device;
        let swapchain = &self.swapchain;

        let swapchain_image = swapchain.acquire_next_image();
        if swapchain_image.is_empty() {
            return;
        }

        let mut command_list = device.create_command_list(daxa_rs::command_list::CommandListInfo {
            debug_name: format!("{} command_list", APPNAME_PREFIX).into(),
        }).unwrap();

        // Set up resources
        use daxa_rs::gpu_resources::BufferInfo;
        use daxa_rs::gpu_resources::MemoryLocation;
        use std::mem::size_of;

        let vertex_staging_buffer = device.create_buffer(BufferInfo {
            memory_location: MemoryLocation::CpuToGpu,
            size: MAX_VERTS * size_of::<DrawVertex>() as u32,
            debug_name: format!("{} vertex_staging_buffer", APPNAME_PREFIX).into()
        }).unwrap();
        command_list.destroy_buffer_deferred(vertex_staging_buffer);

        let buffer_slice = device.get_host_mapped_slice_mut::<DrawVertex>(vertex_staging_buffer).unwrap();
        self.construct_scene(buffer_slice);

        command_list.pipeline_barrier(daxa_rs::split_barrier::MemoryBarrierInfo {
            awaited_pipeline_access: daxa_rs::types::access_consts::HOST_WRITE,
            waiting_pipeline_access: daxa_rs::types::access_consts::TRANSFER_READ,
        });

        command_list.copy_buffer_to_buffer(daxa_rs::command_list::BufferCopyInfo {
            src_buffer: vertex_staging_buffer,
            dst_buffer: self.vertex_buffer,
            size: (self.vertex_count.get() * size_of::<DrawVertex>() as u32) as u64,
            ..Default::default()
        });

        command_list.pipeline_barrier(daxa_rs::split_barrier::MemoryBarrierInfo {
            awaited_pipeline_access: daxa_rs::types::access_consts::TRANSFER_WRITE,
            waiting_pipeline_access: daxa_rs::types::access_consts::VERTEX_SHADER_READ,
        });

        command_list.pipeline_barrier_image_transition(daxa_rs::split_barrier::ImageBarrierInfo {
            waiting_pipeline_access: daxa_rs::types::access_consts::COLOR_ATTACHMENT_OUTPUT_WRITE,
            before_layout: vk::ImageLayout::UNDEFINED,
            after_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL,
            image: swapchain_image,
            ..Default::default()
        });

        // Set up rendering
        use daxa_rs::command_list::RenderAttachmentInfo;
        use daxa_rs::command_list::AttachmentLoadOp;

        let window_size = self.window.get_size();
        command_list.begin_renderpass(daxa_rs::command_list::RenderPassBeginInfo {
            color_attachments: vec![
                RenderAttachmentInfo {
                    image_view: swapchain_image.default_view(),
                    load_op: AttachmentLoadOp::CLEAR,
                    clear_value: vk::ClearValue { color: vk::ClearColorValue { float32: [0.5, 0.5, 0.5, 1.0] } },
                    ..Default::default()
                }
            ],
            render_area: daxa_rs::types::Rect2D {
                offset: daxa_rs::types::Offset2D { x: 0, y: 0 },
                extent: daxa_rs::types::Extent2D { width: window_size.0 as u32, height: window_size.1 as u32 }
            },
            ..Default::default()
        });

        command_list.set_raster_pipeline(self.raster_pipeline.clone());

        command_list.push_constant::<DrawPush>(&DrawPush {
                face_buffer: self.vertex_buffer.index()
            },
            0
        );

        command_list.draw(daxa_rs::command_list::DrawInfo {
            vertex_count: self.vertex_count.get(),
            ..Default::default()
        });

        command_list.end_renderpass();

        // IMGUI record commands

        // Finalize command list
        command_list.pipeline_barrier_image_transition(daxa_rs::split_barrier::ImageBarrierInfo {
            waiting_pipeline_access: daxa_rs::types::access_consts::ALL_GRAPHICS_READ_WRITE,
            before_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL,
            after_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            image: swapchain_image,
            ..Default::default()
        });

        let command_list = command_list.complete();

        device.submit_commands(daxa_rs::device::CommandSubmitInfo {
            command_lists: vec![command_list],
            wait_binary_semaphores: vec![swapchain.get_acquire_semaphore()],
            signal_binary_semaphores: vec![swapchain.get_present_semaphore()],
            signal_timeline_semaphores: vec![(swapchain.get_gpu_timeline_semaphore(), swapchain.get_cpu_timeline_value() as u64)],
            ..Default::default()
        });

        device.preset_frame(daxa_rs::device::PresentInfo {
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

impl Drop for App {
    fn drop(&mut self) {
        self.device.wait_idle();
        self.device.collect_garbage();
        self.device.destroy_buffer(self.vertex_buffer);
    }
}

