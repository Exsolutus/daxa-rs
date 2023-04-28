use crate::device::*;

use anyhow::{Context, Result, bail};
use ash::vk::{self, Offset2D, Extent2D};

use std::{
    borrow::Cow,
    ffi::{
        CStr,
        CString
    },
    slice,
    rc::Rc,
    sync::{
        atomic::Ordering,
        Arc,
    }
};

// Reexport
pub use vk::{
    Format,
    CompareOp,
    PrimitiveTopology,
    PolygonMode,
    CullModeFlags,
    FrontFace,
    PipelineColorBlendAttachmentState,
    BlendFactor,
    BlendOp,
    ColorComponentFlags
};
#[cfg(feature = "conservative_rasterization")]
pub use vk::{
    ConservativeRasterizationModeEXT as ConservativeRasterizationMode
};



pub type ShaderBinary = Cow<'static, Vec<u32>>;

pub(crate) const PIPELINE_MANAGER_MAX_ATTACHMENTS: usize = 16;



pub(crate) struct PipelineZombie {
    pub pipeline: vk::Pipeline
}


#[derive(Default)]
pub struct ShaderInfo {
    pub binary: ShaderBinary,
    pub entry_point: CString
}

pub struct ComputePipelineInfo {
    pub shader_info: ShaderInfo,
    pub push_constant_index: u32,
    pub debug_name: Cow<'static, str>
}

#[derive(Clone)]
pub struct ComputePipeline(pub(crate) Rc<ComputePipelineInternal>);

pub(crate) struct ComputePipelineInternal {
    device: Device,
    pub info: ComputePipelineInfo,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout
}

// ComputePipeline creation methods
impl ComputePipeline {
    pub(crate) fn new(device: Device, info: ComputePipelineInfo) -> Result<Self> {
        let shader_module_ci = vk::ShaderModuleCreateInfo::builder()
            .code(&info.shader_info.binary)
            .build();

        let shader_module = unsafe {
            device.0.logical_device.create_shader_module(&shader_module_ci, None)
                .context("ShaderModule should be created.")?
        };

        let pipeline_layout = device.0.gpu_shader_resource_table.pipeline_layouts[info.push_constant_index as usize];
        
        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&info.shader_info.entry_point)
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
                .object_type(vk::ObjectType::PIPELINE)
                .object_handle(vk::Handle::as_raw(pipeline))
                .object_name(&CStr::from_ptr(pipeline_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &pipeline_name_info)?;
        }

        Ok(Self(Rc::new(ComputePipelineInternal {
            device,
            info,
            pipeline: pipeline.into(),
            pipeline_layout
        })))
    }
}

// ComputePipeline usage methods
impl ComputePipeline {
    #[inline]
    pub fn info(&self) -> &ComputePipelineInfo {
        &self.0.info
    }
}

// ComputePipeline internal methods
impl Drop for ComputePipelineInternal {
    fn drop(&mut self) {
        let mut zombies = self.device.0.main_queue_zombies.lock().unwrap();
        let cpu_timeline_value = self.device.0.main_queue_cpu_timeline.load(Ordering::Acquire);

        zombies.pipelines.push_front((
            cpu_timeline_value,
            PipelineZombie {
                pipeline: self.pipeline
            }
        ))
    }
}


#[derive(Clone, Copy)]
pub struct DepthTestInfo {
    pub depth_attachment_format: Format,
    pub enable_depth_test: bool,
    pub enable_depth_write: bool,
    pub depth_test_compare_op: CompareOp,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32
}

impl Default for DepthTestInfo {
    fn default() -> Self {
        Self {
            depth_attachment_format: Format::UNDEFINED,
            enable_depth_test: false,
            enable_depth_write: false,
            depth_test_compare_op: CompareOp::LESS_OR_EQUAL,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0
        }
    }
}

#[cfg(feature = "conservative_rasterization")]
#[derive(Clone, Copy, Default)]
pub struct ConservativeRasterInfo {
    pub mode: ConservativeRasterizationMode,
    pub size: f32
}

#[derive(Clone, Copy)]
pub struct RasterizerInfo {
    pub primitive_topology: PrimitiveTopology,
    pub primitive_restart_enable: bool,
    pub polygon_mode: PolygonMode,
    pub face_culling: CullModeFlags,
    pub front_face_winding: FrontFace,
    pub rasterizer_discard_enable: bool,
    pub depth_clamp_enable: bool,
    pub depth_bias_enable: bool,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
    #[cfg(feature = "conservative_rasterization")]
    pub conservative_raster_info: ConservativeRasterInfo
}

impl Default for RasterizerInfo {
    fn default() -> Self {
        Self {
            primitive_topology: PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false,
            polygon_mode: PolygonMode::FILL,
            face_culling: CullModeFlags::NONE,
            front_face_winding: FrontFace::CLOCKWISE,
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
            #[cfg(feature = "conservative_rasterization")]
            conservative_raster_info: ConservativeRasterInfo::default()
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct RenderAttachment {
    pub format: Format,
    pub blend: PipelineColorBlendAttachmentState
}

#[derive(Default)]
pub struct RasterPipelineInfo {
    pub vertex_shader_info: ShaderInfo,
    pub fragment_shader_info: ShaderInfo,
    pub color_attachments: Vec<RenderAttachment>,
    pub depth_test: DepthTestInfo,
    pub raster: RasterizerInfo,
    pub push_constant_index: u32,
    pub debug_name: Cow<'static, str>
}

#[derive(Clone)]
pub struct RasterPipeline(pub(crate) Arc<RasterPipelineInternal>);

pub(crate) struct RasterPipelineInternal {
    device: Device,
    pub info: RasterPipelineInfo,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout
}

// RasterPipeline creation methods
impl RasterPipeline {
    pub(crate) fn new(device: Device, info: RasterPipelineInfo) -> Result<Self> {
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
                .name(&info.entry_point)
                .build();
            pipeline_shader_state_create_infos.push(shader_stage_ci);

            Ok(())
        };

        create_shader_module(&info.vertex_shader_info, vk::ShaderStageFlags::VERTEX)?;
        create_shader_module(&info.fragment_shader_info, vk::ShaderStageFlags::FRAGMENT)?;
        
        let pipeline_layout = device.0.gpu_shader_resource_table.pipeline_layouts[info.push_constant_index as usize];

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
        
        const DEFAULT_VIEWPORT: vk::Viewport = vk::Viewport { x: 0.0, y: 0.0, width: 1.0, height: 1.0, min_depth: 0.0, max_depth: 0.0 };
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
                .object_type(vk::ObjectType::PIPELINE)
                .object_handle(vk::Handle::as_raw(pipeline))
                .object_name(&CStr::from_ptr(pipeline_name.as_ptr() as *const i8));
            device.debug_utils().set_debug_utils_object_name(device.0.logical_device.handle(), &pipeline_name_info)?;
        }


        Ok(Self(Arc::new(RasterPipelineInternal {
            device,
            info,
            pipeline,
            pipeline_layout
        })))
    }
}

// RasterPipeline usage methods
impl RasterPipeline {
    #[inline]
    pub fn info(&self) -> &RasterPipelineInfo {
        &self.0.info
    }
}

// RasterPipeline internal methods
impl Drop for RasterPipelineInternal {
    fn drop(&mut self) {
        let mut zombies = self.device.0.main_queue_zombies.lock().unwrap();
        let cpu_timeline_value = self.device.0.main_queue_cpu_timeline.load(Ordering::Acquire);

        zombies.pipelines.push_front((
            cpu_timeline_value,
            PipelineZombie {
                pipeline: self.pipeline
            }
        ))
    }
}