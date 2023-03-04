use crate::{core::*, device::*};

use anyhow::{Context, Result, bail};
use ash::vk::{self, Offset2D, Extent2D};

use std::{
    borrow::Cow,
    ffi::{
        CStr,
        CString
    },
    mem::size_of,
    slice
};


pub type ShaderBinary = Vec<u32>;

pub(crate) const PIPELINE_MANAGER_MAX_ATTACHMENTS: usize = 16;



pub(crate) struct PipelineZombie {
    pipeline: vk::Pipeline
}



pub struct ShaderInfo {
    binary: ShaderBinary,
    entry_point: Option<&'static str>
}

pub struct ComputePipelineInfo {
    shader_info: ShaderInfo,
    push_constant_size: u32,
    debug_name: Cow<'static, str>
}

pub struct ComputePipeline {
    device: Device,
    info: ComputePipelineInfo,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout
}

impl ComputePipeline {
    pub fn new(device: Device, info: ComputePipelineInfo) -> Result<Self> {
        let shader_module_ci = vk::ShaderModuleCreateInfo::builder()
            .code(&info.shader_info.binary)
            .build();

        let shader_module = unsafe {
            device.0.logical_device.create_shader_module(&shader_module_ci, None)
                .context("ShaderModule should be created.")?
        };

        let pipeline_layout = device.0.gpu_shader_resource_table.pipeline_layouts[((info.push_constant_size + 3) / 4) as usize];
        
        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&CString::new(info.shader_info.entry_point.unwrap_or("main")).unwrap())
                .build()
            )
            .layout(pipeline_layout);

        let pipeline = unsafe {
            match device.0.logical_device.create_compute_pipelines(vk::PipelineCache::null(), slice::from_ref(&compute_pipeline_create_info), None) {
                Ok(result) => result[0],
                Err((_, error)) => bail!(error)
            }
        };

        unsafe { device.0.logical_device.destroy_shader_module(shader_module, None) };

        #[cfg(debug_assertions)]
        unsafe {
            let pipeline_name = format!("{} [Daxa ComputePipeline]\0", info.debug_name);
            let pipeline_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::SEMAPHORE)
                .object_handle(vk::Handle::as_raw(pipeline))
                .object_name(&CStr::from_ptr(pipeline_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &pipeline_name_info)?;
        }

        Ok(Self {
            device,
            info,
            pipeline,
            pipeline_layout
        })
    }

    #[inline]
    pub fn info(&self) -> &ComputePipelineInfo {
        &self.info
    }
}



pub struct DepthTestInfo {
    depth_attachment_format: vk::Format,
    enable_depth_test: bool,
    enable_depth_write: bool,
    depth_test_compare_op: vk::CompareOp,
    min_depth_bounds: f32,
    max_depth_bounds: f32
}

#[cfg(feature = "conservative_rasterization")]
pub struct ConservativeRasterInfo {
    mode: vk::ConservativeRasterizationModeEXT,
    size: f32
}

pub struct RasterizerInfo {
    primitive_topology: vk::PrimitiveTopology,
    primitive_restart_enable: bool,
    polygon_mode: vk::PolygonMode,
    face_culling: vk::CullModeFlags,
    front_face_winding: vk::FrontFace,
    rasterizer_discard_enable: bool,
    depth_clamp_enable: bool,
    depth_bias_enable: bool,
    depth_bias_constant_factor: f32,
    depth_bias_clamp: f32,
    depth_bias_slope_factor: f32,
    line_width: f32,
    #[cfg(feature = "conservative_rasterization")]
    conservative_raster_info: ConservativeRasterInfo
}

pub struct RenderAttachment {
    format: vk::Format,
    blend: vk::PipelineColorBlendAttachmentState
}

pub struct RasterPipelineInfo {
    vertex_shader_info: ShaderInfo,
    fragment_shader_info: ShaderInfo,
    color_attachments: Vec<RenderAttachment>,
    depth_test: DepthTestInfo,
    raster: RasterizerInfo,
    push_constant_size: u32,
    debug_name: Cow<'static, str>
}

pub struct RasterPipeline {
    device: Device,
    info: RasterPipelineInfo,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout
}

impl RasterPipeline {
    pub fn new(device: Device, info: RasterPipelineInfo) -> Result<Self> {
        let mut shader_modules = vec![];
        let mut pipeline_shader_state_create_infos = vec![];

        let mut create_shader_module = |info: &ShaderInfo, stage: vk::ShaderStageFlags| -> Result<()> {
            let shader_module_ci = vk::ShaderModuleCreateInfo::builder()
                .code(&info.binary)
                .build();
            let shader_module = unsafe {
                device.0.logical_device.create_shader_module(&shader_module_ci, None)
                    .context("ShaderModule should be created.")?
            };
            shader_modules.push(shader_module);

            let shader_stage_ci = vk::PipelineShaderStageCreateInfo::builder()
                .stage(stage)
                .module(shader_module)
                .name(&CString::new(info.entry_point.unwrap_or("main")).unwrap())
                .build();
            pipeline_shader_state_create_infos.push(shader_stage_ci);

            Ok(())
        };

        create_shader_module(&info.vertex_shader_info, vk::ShaderStageFlags::VERTEX)?;
        create_shader_module(&info.fragment_shader_info, vk::ShaderStageFlags::FRAGMENT)?;
        
        let pipeline_layout = device.0.gpu_shader_resource_table.pipeline_layouts[((info.push_constant_size + 3) / 4) as usize];

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(info.raster.primitive_topology)
            .primitive_restart_enable(info.raster.primitive_restart_enable);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .min_sample_shading(1.0);

        let raster_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(info.raster.polygon_mode)
            .cull_mode(info.raster.face_culling)
            .front_face(info.raster.front_face_winding)
            .depth_bias_enable(info.raster.depth_bias_enable)
            .depth_bias_constant_factor(info.raster.depth_bias_constant_factor)
            .depth_bias_clamp(info.raster.depth_bias_clamp)
            .depth_bias_slope_factor(info.raster.depth_bias_slope_factor)
            .line_width(info.raster.line_width);

        #[cfg(feature = "conservative_rasterization")]
        let raster_state = {
            let mut conservative_raster_state = vk::PipelineRasterizationConservativeStateCreateInfoEXT::builder()
                .conservative_rasterization_mode(info.raster.conservative_raster_info.mode)
                .extra_primitive_overestimation_size(info.raster.conservative_raster_info.size)
                .build();
            raster_state.push_next(&mut conservative_raster_state).build()
        };
        #[cfg(not(feature = "conservative_rasterization"))]
        let raster_state = raster_state.build();

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(info.depth_test.enable_depth_test)
            .depth_write_enable(info.depth_test.enable_depth_write)
            .depth_compare_op(info.depth_test.depth_test_compare_op)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .min_depth_bounds(info.depth_test.min_depth_bounds)
            .max_depth_bounds(info.depth_test.max_depth_bounds);

        debug_assert!(info.color_attachments.len() < PIPELINE_MANAGER_MAX_ATTACHMENTS, "Too many color attachments. Make a pull request to bump max.");
        
        let mut pipeline_color_blend_attachment_blend_states = vec![];
        let mut pipeline_color_attachment_formats = vec![];
        
        for attachment in &info.color_attachments {
            pipeline_color_blend_attachment_blend_states.push(attachment.blend);
            pipeline_color_attachment_formats.push(attachment.format);
        }

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&pipeline_color_blend_attachment_blend_states)
            .blend_constants([1.0, 1.0, 1.0, 1.0]);
        
        const DEFAULT_VIEWPORT: vk::Viewport = vk::Viewport { x: 0f32, y: 0f32, width: 1f32, height: 1f32, min_depth: 0f32, max_depth: 0f32 };
        const DEFAULT_SCISSOR: vk::Rect2D = vk::Rect2D { offset: Offset2D { x: 0, y: 0 }, extent: Extent2D { width: 1, height: 1 } };

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(slice::from_ref(&DEFAULT_VIEWPORT))
            .scissors(slice::from_ref(&DEFAULT_SCISSOR));

        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::DEPTH_BIAS
        ];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);

        let mut pipeline_rendering = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&pipeline_color_attachment_formats)
            .depth_attachment_format(info.depth_test.depth_attachment_format)
            .build();

        let graphics_pipeline_ci = vk::GraphicsPipelineCreateInfo::builder()
            .push_next(&mut pipeline_rendering)
            .stages(&pipeline_shader_state_create_infos)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .build();

        let pipeline = unsafe {
            match device.0.logical_device.create_graphics_pipelines(vk::PipelineCache::null(), slice::from_ref(&graphics_pipeline_ci), None) {
                Ok(result) => result[0],
                Err((_, error)) => bail!(error)
            }
        };

        for module in shader_modules {
            unsafe { device.0.logical_device.destroy_shader_module(module, None) }
        }

        #[cfg(debug_assertions)]
        unsafe {
            let pipeline_name = format!("{} [Daxa RasterPipeline]\0", info.debug_name);
            let pipeline_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::SEMAPHORE)
                .object_handle(vk::Handle::as_raw(pipeline))
                .object_name(&CStr::from_ptr(pipeline_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &pipeline_name_info)?;
        }


        Ok(Self {
            device,
            info,
            pipeline,
            pipeline_layout
        })
    }

    #[inline]
    pub fn info(&self) -> &RasterPipelineInfo {
        &self.info
    }
}