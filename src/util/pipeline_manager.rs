use crate::{core::*, device::*, pipeline::*};

use anyhow::{Context as _, Result, bail};

use std::{
    borrow::Cow,
    collections::HashMap, 
    path::{
        Path,
        PathBuf
    },
    sync::{
        Arc,
        Mutex
    }
};



pub struct ShaderCrate(pub PathBuf);

pub struct ShaderCode(pub String);

pub enum ShaderSource {
    Crate(ShaderCrate),
    Code(ShaderCode),
    Binary(ShaderBinary)
}

impl From<ShaderCrate> for ShaderSource {
    fn from(value: ShaderCrate) -> Self {
        Self::Crate(value)
    }
}

impl From<ShaderCode> for ShaderSource {
    fn from(value: ShaderCode) -> Self {
        Self::Code(value)
    }
}

impl From<ShaderBinary> for ShaderSource {
    fn from(value: ShaderBinary) -> Self {
        Self::Binary(value)
    }
}

pub struct ShaderDefine {
    pub name: Cow<'static, str>,
    pub value: Cow<'static, str>
}

pub struct ShaderCompileOptions {
    pub entry_point: &'static str,
    pub root_paths: Vec<PathBuf>,
    pub write_out_preprocessed_code: Option<PathBuf>,
    pub write_out_shader_binary: Option<PathBuf>,
    //pub defines: Vec<ShaderDefine>
}

impl Default for ShaderCompileOptions {
    fn default() -> Self {
        Self {
            entry_point: "main",
            root_paths: vec![],
            write_out_preprocessed_code: None,
            write_out_shader_binary: None,
            //defines: vec![]
        }
    }
}

pub struct ShaderCompileInfo {
    pub source: ShaderSource,
    pub compile_options: ShaderCompileOptions
}

pub struct ComputePipelineCompileInfo {
    pub shader_info: ShaderCompileInfo,
    pub push_constant_index: u32,
    pub debug_name: Cow<'static, str>
}

pub struct RasterPipelineCompileInfo {
    pub vertex_shader_info: ShaderCompileInfo,
    pub fragment_shader_info: ShaderCompileInfo,
    pub color_attachments: Vec<RenderAttachment>,
    pub depth_test: DepthTestInfo,
    pub raster: RasterizerInfo,
    pub push_constant_index: u32,
    pub debug_name: Cow<'static, str>
}

pub struct PipelineManagerInfo {
    pub device: Device,
    pub shader_compile_options: ShaderCompileOptions,
    pub debug_name: Cow<'static, str>
}



type ShaderCrateSourceCache = Arc<Mutex<HashMap<PathBuf, ShaderBinary>>>;

enum ShaderStage {
    Compute,
    Vertex,
    Fragment
}

struct PipelineState<PipeT, InfoT> {
    pipeline: PipeT,
    info: InfoT,
    //observed_hotload_crates: ShaderCrateSourceCache
}

type ComputePipelineState = PipelineState<ComputePipeline, ComputePipelineCompileInfo>;
type RasterPipelineState = PipelineState<RasterPipeline, RasterPipelineCompileInfo>;



pub struct PipelineManager {
    info: PipelineManagerInfo,
    current_seen_shader_crates: Vec<PathBuf>,
    current_observed_hotload_crates: ShaderCrateSourceCache,

    compute_pipelines: Vec<ComputePipelineState>,
    raster_pipelines: Vec<RasterPipelineState>,
}

// PipelineManager creation methods
impl PipelineManager {
    pub fn new(info: PipelineManagerInfo) -> Result<Self> {
        Ok(Self {
            info,
            current_seen_shader_crates: vec![],
            current_observed_hotload_crates: Default::default(),
            compute_pipelines: vec![],
            raster_pipelines: vec![],
        })
    }
}

// PipelineManager usage methods
impl PipelineManager {
    pub fn add_compute_pipeline(&mut self, info: ComputePipelineCompileInfo) -> Result<ComputePipeline> {
        let result = self.create_compute_pipeline(info)?;
        let pipeline = result.pipeline.clone();

        self.compute_pipelines.push(result);

        Ok(pipeline)
    }

    pub fn add_raster_pipeline(&mut self, info: RasterPipelineCompileInfo) -> Result<RasterPipeline> {
        let result = self.create_raster_pipeline(info)?;
        let pipeline = result.pipeline.clone();

        self.raster_pipelines.push(result);

        Ok(pipeline)
    }

    pub fn remove_compute_pipeline(&self, pipeline: ComputePipeline) {
        todo!()
    }

    pub fn remove_raster_pipeline(&self, pipeline: RasterPipeline) {
        todo!()
    }

    pub fn reload_all(&self) -> Result<bool> {
        todo!()
    }
}

// PipelineManager internal methods
impl PipelineManager {
    fn create_compute_pipeline(&self, info: ComputePipelineCompileInfo) -> Result<ComputePipelineState> {
        if info.push_constant_index > PIPELINE_LAYOUT_COUNT {
            bail!("Push constant index {} exceeds maximum index of {}.", info.push_constant_index, PIPELINE_LAYOUT_COUNT)
        }

        let spirv = match &info.shader_info.source {
            ShaderSource::Crate(path) => {
                let full_path = self.full_path_to_crate(&info.shader_info)
                    .context(format!("Shader crate not found in any known root with path: {:?}", path.0))?;

                match self.load_shader_crate(&full_path) {
                    Ok(spirv) => spirv.clone(),
                    Err(error) => bail!(error)
                }
            },
            ShaderSource::Code(_) => {
                todo!()
            },
            ShaderSource::Binary(spirv) => {
                spirv.clone()
            }
        };

        let pipeline = self.info.device.create_compute_pipeline(ComputePipelineInfo {
            shader_info: ShaderInfo {
                binary: spirv.clone(),
                entry_point: std::ffi::CString::new(info.shader_info.compile_options.entry_point).unwrap()
            },
            push_constant_index: info.push_constant_index,
            debug_name: info.debug_name.clone()
        })?;

        Ok(ComputePipelineState {
            info,
            pipeline,
        })
    }

    fn create_raster_pipeline(&self, info: RasterPipelineCompileInfo) -> Result<RasterPipelineState> {
        if info.push_constant_index > PIPELINE_LAYOUT_COUNT {
            bail!("Push constant index {} exceeds maximum index of {}.", info.push_constant_index, PIPELINE_LAYOUT_COUNT)
        }

        let vertex_spirv = match &info.vertex_shader_info.source {
            ShaderSource::Crate(path) => {
                let full_path = self.full_path_to_crate(&info.vertex_shader_info)
                    .context(format!("Shader crate not found in any known root with path: {:?}", path.0))?;

                match self.load_shader_crate(&full_path) {
                    Ok(spirv) => spirv.clone(),
                    Err(error) => bail!(error)
                }
            },
            ShaderSource::Code(_) => {
                todo!()
            },
            ShaderSource::Binary(spirv) => {
                spirv.clone()
            }
        };

        let fragment_spirv = match &info.fragment_shader_info.source {
            ShaderSource::Crate(path) => {
                let full_path = self.full_path_to_crate(&info.fragment_shader_info)
                    .context(format!("Shader crate not found in any known root with path: {:?}", path.0))?;

                match self.load_shader_crate(&full_path) {
                    Ok(spirv) => spirv.clone(),
                    Err(error) => bail!(error)
                }
            },
            ShaderSource::Code(_) => {
                todo!()
            },
            ShaderSource::Binary(spirv) => {
                spirv.clone()
            }
        };

        let pipeline = self.info.device.create_raster_pipeline(RasterPipelineInfo {
            vertex_shader_info: ShaderInfo {
                binary: vertex_spirv.clone(),
                entry_point: std::ffi::CString::new(info.vertex_shader_info.compile_options.entry_point).unwrap()
            },
            fragment_shader_info: ShaderInfo {
                binary: fragment_spirv.clone(),
                entry_point: std::ffi::CString::new(info.fragment_shader_info.compile_options.entry_point).unwrap()
            },
            color_attachments: info.color_attachments.clone(),
            depth_test: info.depth_test,
            raster: info.raster,
            push_constant_index: info.push_constant_index,
            debug_name: info.debug_name.clone()
        })?;

        Ok(RasterPipelineState {
            info,
            pipeline
        })
    }


    fn load_shader_crate(&self, path: &Path) -> Result<ShaderBinary> {
        use spirv_builder::{CompileResult, MetadataPrintout, SpirvBuilder, Capability};

        let crate_path: PathBuf = path.into();

        if let Some(spirv) = self.current_observed_hotload_crates.lock().unwrap().get(&crate_path) {
            return Ok(spirv.clone())
        }

        // Hack: spirv_builder builds into a custom directory if running under cargo, to not
        // deadlock, and the default target directory if not. However, packages like `proc-macro2`
        // have different configurations when being built here vs. when building
        // rustc_codegen_spirv normally, so we *want* to build into a separate target directory, to
        // not have to rebuild half the crate graph every time we run. So, pretend we're running
        // under cargo by setting these environment variables.
        std::env::set_var("OUT_DIR", env!("OUT_DIR"));
        std::env::set_var("PROFILE", env!("PROFILE"));

        let builder = SpirvBuilder::new(&crate_path, "spirv-unknown-vulkan1.2")
            .print_metadata(MetadataPrintout::None)
            .capability(Capability::RuntimeDescriptorArray)
            .extension("SPV_EXT_descriptor_indexing")
            .preserve_bindings(true);

        let initial_result = {
            let crate_path: PathBuf = path.into();
            let observed_crates = self.current_observed_hotload_crates.clone();
            builder.watch(move |compile_result| {
                let mut lock = observed_crates.lock().unwrap();
                let Some(cached_result) = lock.get_mut(&crate_path) else {
                    panic!("Attempt to hotreload an untracked shader crate.");
                };
                *cached_result = handle_compile_result(compile_result).unwrap();
            }).expect("Builder should watch for source changes.")
            //builder.build()?
        };

        fn handle_compile_result(compile_result: CompileResult) -> Result<ShaderBinary> {
            let module_path = compile_result.module.unwrap_single();
            let bytes = &mut std::fs::File::open(module_path).unwrap();
            let spirv = ash::util::read_spv(bytes)
                .context("Shader crate should be compiled to SPIR-V.")?;

            Ok(ShaderBinary::from(Cow::Owned(spirv.into())))
        }

        let source = handle_compile_result(initial_result)?;

        let mut lock = self.current_observed_hotload_crates.lock().unwrap();
        lock.insert(
            crate_path, 
            source.clone()
        );

        Ok(source)
    }

    fn full_path_to_crate(&self, info: &ShaderCompileInfo) -> Result<PathBuf> {
        let ShaderSource::Crate(path) = &info.source else {
            bail!("ShaderCompileInfo must have source type ShaderSource::Crate.")
        };

        if path.0.is_dir() {
            return Ok(path.0.clone())
        }

        let full_path = info.compile_options.root_paths.iter().find_map(|root| {
            let shader_crate = [root, &path.0]
                .iter()
                .copied()
                .collect::<PathBuf>();
            match shader_crate.is_dir() {
                true => Some(shader_crate),
                false => None
            }
        })
        .or_else(|| {
            self.info.shader_compile_options.root_paths.iter().find_map(|root| {
                let shader_crate = [root, &path.0]
                    .iter()
                    .copied()
                    .collect::<PathBuf>();
                match shader_crate.is_dir() {
                    true => Some(shader_crate),
                    false => None
                }
            })
        });

        match full_path {
            Some(path) => Ok(path),
            None => bail!("")
        }
    }
}