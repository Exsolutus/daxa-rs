use crate::{core::*, device::*, pipeline::*};

use anyhow::{Context as _, Result, bail};

use std::{
    borrow::Cow,
    cell::UnsafeCell,
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



#[derive(Clone, Copy)]
pub struct PipelineId(u32); // TODO: maybe have ID type per pipeline type



pub type ShaderCrate = PathBuf;

pub type ShaderCode = String;

#[derive(Clone)]
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



#[derive(Clone)]
pub struct ShaderCompileOptions {
    pub entry_point: &'static str,
    pub root_paths: Vec<PathBuf>,
    pub write_out_preprocessed_code: Option<PathBuf>,
    pub write_out_shader_binary: Option<PathBuf>,
}

impl Default for ShaderCompileOptions {
    fn default() -> Self {
        Self {
            entry_point: "main",
            root_paths: vec![],
            write_out_preprocessed_code: None,
            write_out_shader_binary: None,
        }
    }
}

#[derive(Clone)]
pub struct ShaderCompileInfo {
    pub source: ShaderSource,
    pub compile_options: ShaderCompileOptions
}

#[derive(Clone)]
pub struct ComputePipelineCompileInfo {
    pub shader_info: ShaderCompileInfo,
    pub push_constant_index: u32,
    pub debug_name: Cow<'static, str>
}

#[derive(Clone)]
pub struct RasterPipelineCompileInfo {
    pub vertex_shader_info: ShaderCompileInfo,
    pub fragment_shader_info: ShaderCompileInfo,
    pub color_attachments: Vec<RenderAttachment>,
    pub depth_test: DepthTestInfo,
    pub raster: RasterizerInfo,
    pub push_constant_index: u32,
    pub debug_name: Cow<'static, str>
}

type ShaderCrateSourceCache = HashMap<PathBuf, ShaderBinary>;



#[derive(Clone)]
struct PipelineState<PipeT, InfoT> {
    pipeline: PipeT,
    info: InfoT,
}

type ComputePipelineState = PipelineState<ComputePipeline, ComputePipelineCompileInfo>;
type RasterPipelineState = PipelineState<RasterPipeline, RasterPipelineCompileInfo>;

struct PipelinePool<T> {
    pipelines: UnsafeCell<Vec<Option<T>>>,
    free_index_stack: UnsafeCell<Vec<u32>>,
}

type ComputePipelinePool = PipelinePool<ComputePipelineState>;
type RasterPipelinePool = PipelinePool<RasterPipelineState>;

impl<T> Default for PipelinePool<T> {
    fn default() -> Self {
        Self {
            pipelines: Default::default(),
            free_index_stack: Default::default()
        }
    }
}

impl<T> PipelinePool<T> {
    fn push(&self, value: T) -> PipelineId {
        let pipelines = unsafe { self.pipelines.get().as_mut().unwrap_unchecked() };
        let free_index_stack = unsafe { self.free_index_stack.get().as_mut().unwrap_unchecked() };
        match free_index_stack.pop() {
            Some(id) => {
                pipelines[id as usize] = Some(value);

                PipelineId(id)
            },
            None => {
                let id = PipelineId(pipelines.len() as u32);
                pipelines.push(Some(value));
        
                id
            }
        }
    }

    fn get(&self, id: PipelineId) -> Result<&T> {
        let pipelines = unsafe { self.pipelines.get().as_ref().unwrap_unchecked() };
        match &pipelines[id.0 as usize] {
            Some(pipeline) => {
                Ok(&pipeline)
            },
            None => {
                bail!("PipelineId {} is invalid.", id.0);
            }
        }
    }

    fn iter(&self) -> Vec<&T> {
        let pipelines = unsafe { self.pipelines.get().as_ref().unwrap_unchecked() };

        pipelines.iter().filter_map(|value| value.as_ref()).collect::<Vec<&T>>()
    }

    fn iter_mut(&self) -> Vec<&mut T> {
        let pipelines = unsafe { self.pipelines.get().as_mut().unwrap_unchecked() };

        pipelines.iter_mut().filter_map(|value| value.as_mut()).collect::<Vec<&mut T>>()
    }

    fn remove(&self, id: PipelineId) -> Result<T> {
        let pipelines = unsafe { self.pipelines.get().as_mut().unwrap_unchecked() };
        let free_index_stack = unsafe { self.free_index_stack.get().as_mut().unwrap_unchecked() };
        match pipelines[id.0 as usize].take() {
            Some(pipeline) => {
                free_index_stack.push(id.0);
                Ok(pipeline)
            },
            None => {
                bail!("PipelineId {} is invalid.", id.0);
            }
        }
    }
}



pub struct PipelineManagerInfo {
    pub device: Device,
    pub shader_compile_options: ShaderCompileOptions,
    pub debug_name: Cow<'static, str>
}

pub struct PipelineManager {
    info: PipelineManagerInfo,
    source_cache: UnsafeCell<ShaderCrateSourceCache>,
    hot_reload_source_cache: Arc<Mutex<ShaderCrateSourceCache>>,

    compute_pipelines: ComputePipelinePool,
    raster_pipelines: RasterPipelinePool,
}

// PipelineManager creation methods
impl PipelineManager {
    pub fn new(info: PipelineManagerInfo) -> Result<Self> {
        Ok(Self {
            info,
            source_cache: Default::default(),
            hot_reload_source_cache: Default::default(),
            compute_pipelines: Default::default(),
            raster_pipelines: Default::default(),
        })
    }
}

// PipelineManager usage methods
impl PipelineManager {
    pub fn add_compute_pipeline(&self, info: ComputePipelineCompileInfo) -> Result<PipelineId> {
        let result = self.create_compute_pipeline(&info)?;

        Ok(self.compute_pipelines.push(result))
    }

    pub fn add_raster_pipeline(&mut self, info: RasterPipelineCompileInfo) -> Result<PipelineId> {
        let result = self.create_raster_pipeline(&info)?;

        Ok(self.raster_pipelines.push(result))
    }

    pub fn get_compute_pipeline(&self, id: PipelineId) -> Result<&ComputePipeline> {
        Ok(&self.compute_pipelines.get(id)?.pipeline)
    }

    pub fn get_raster_pipeline(&self, id: PipelineId) -> Result<&RasterPipeline> {
        Ok(&self.raster_pipelines.get(id)?.pipeline)
    }

    pub fn remove_compute_pipeline(&self, id: PipelineId) -> Result<ComputePipeline> {
        Ok(self.compute_pipelines.remove(id)?.pipeline)
    }

    pub fn remove_raster_pipeline(&self, id: PipelineId) -> Result<RasterPipeline> {
        Ok(self.raster_pipelines.remove(id)?.pipeline)
    }

    pub fn reload_all(&self) -> Result<()> {
        let source_cache = unsafe { self.source_cache.get().as_mut().unwrap_unchecked() };
        let mut hot_reload_source_cache = self.hot_reload_source_cache.lock().unwrap();

        for pipeline in self.compute_pipelines.iter_mut() {
            let ShaderSource::Crate(path) = &pipeline.info.shader_info.source else {
                continue;
            };

            let full_path = self.full_path_to_crate(&pipeline.info.shader_info)?;

            if let Some(spirv) = hot_reload_source_cache.get(&full_path) {
                source_cache.insert(path.clone(), spirv.clone());

                let new_pipeline = self.create_compute_pipeline(&pipeline.info)?;
                
                *pipeline = new_pipeline;
            }
        }

        for pipeline in self.raster_pipelines.iter_mut() {
            let (ShaderSource::Crate(_), ShaderSource::Crate(_)) = (&pipeline.info.vertex_shader_info.source, &pipeline.info.fragment_shader_info.source) else {
                continue;
            };

            let vertex_path = self.full_path_to_crate(&pipeline.info.vertex_shader_info)?;
            let fragment_path = self.full_path_to_crate(&pipeline.info.fragment_shader_info)?;

            let mut reload = false;
            if let Some(vertex_spirv) = hot_reload_source_cache.get(&vertex_path) {
                source_cache.insert(vertex_path, vertex_spirv.clone());
                reload = true;
            }
            if let Some(fragment_spirv) = hot_reload_source_cache.get(&fragment_path) {
                source_cache.insert(fragment_path, fragment_spirv.clone());
                reload = true;
            }
            if reload {
                let new_pipeline = self.create_raster_pipeline(&pipeline.info)?;

                *pipeline = new_pipeline;
            }
        }

        hot_reload_source_cache.clear();

        Ok(())
    }
}

// PipelineManager internal methods
impl PipelineManager {
    fn create_compute_pipeline(&self, info: &ComputePipelineCompileInfo) -> Result<ComputePipelineState> {
        if info.push_constant_index > PIPELINE_LAYOUT_COUNT {
            bail!("Push constant index {} exceeds maximum index of {}.", info.push_constant_index, PIPELINE_LAYOUT_COUNT)
        }

        //let new_info = info.clone();

        let spirv = match &info.shader_info.source {
            ShaderSource::Crate(path) => {
                let full_path = self.full_path_to_crate(&info.shader_info)
                    .context(format!("Shader crate not found in any known root with path: {:?}", path))?;

                match self.load_shader_crate(&full_path) {
                    Ok(spirv) => spirv,
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
                binary: spirv,
                entry_point: std::ffi::CString::new(info.shader_info.compile_options.entry_point).unwrap()
            },
            push_constant_index: info.push_constant_index,
            debug_name: info.debug_name.clone()
        })?;

        Ok(ComputePipelineState {
            info: info.clone(),
            pipeline,
        })
    }

    fn create_raster_pipeline(&self, info: &RasterPipelineCompileInfo) -> Result<RasterPipelineState> {
        if info.push_constant_index > PIPELINE_LAYOUT_COUNT {
            bail!("Push constant index {} exceeds maximum index of {}.", info.push_constant_index, PIPELINE_LAYOUT_COUNT)
        }

        let new_info = info.clone();

        let vertex_spirv = match &info.vertex_shader_info.source {
            ShaderSource::Crate(path) => {
                let full_path = self.full_path_to_crate(&info.vertex_shader_info)
                    .context(format!("Shader crate not found in any known root with path: {:?}", path))?;

                match self.load_shader_crate(&full_path) {
                    Ok(spirv) => spirv,
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
                    .context(format!("Shader crate not found in any known root with path: {:?}", path))?;

                let spirv = {
                    if let ShaderSource::Crate(_) = &info.vertex_shader_info.source {
                        let full_vert_path = unsafe { self.full_path_to_crate(&info.vertex_shader_info).unwrap_unchecked() }; // Vertex path validated above
                        if full_path == full_vert_path {
                            Some(&vertex_spirv)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                match spirv {
                    Some(spirv) => {
                        spirv.clone()
                    },
                    None => {
                        match self.load_shader_crate(&full_path) {
                            Ok(spirv) => spirv,
                            Err(error) => bail!(error)
                        }
                    }
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
                binary: vertex_spirv,
                entry_point: std::ffi::CString::new(info.vertex_shader_info.compile_options.entry_point).unwrap()
            },
            fragment_shader_info: ShaderInfo {
                binary: fragment_spirv,
                entry_point: std::ffi::CString::new(info.fragment_shader_info.compile_options.entry_point).unwrap()
            },
            color_attachments: info.color_attachments.clone(),
            depth_test: info.depth_test,
            raster: info.raster,
            push_constant_index: info.push_constant_index,
            debug_name: info.debug_name.clone()
        })?;

        Ok(RasterPipelineState {
            info: new_info,
            pipeline,
        })
    }

    fn load_shader_crate(
        &self,
        path: &Path,
    ) -> Result<ShaderBinary> {
        use spirv_builder::{CompileResult, MetadataPrintout, SpirvBuilder, Capability};

        let source_cache = unsafe { self.source_cache.get().as_mut().unwrap_unchecked() };

        let crate_path: PathBuf = path.into();

        // Check for known source

        if let Some(spirv) = source_cache.get(&crate_path) {
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
            let hot_reload_source_cache = self.hot_reload_source_cache.clone();
            builder.watch(move |compile_result| {
                let mut lock = hot_reload_source_cache.lock().unwrap();
                lock.insert(crate_path.clone(), handle_compile_result(compile_result).unwrap());
            }).expect("Builder should watch for source changes.")
        };

        fn handle_compile_result(compile_result: CompileResult) -> Result<ShaderBinary> {
            let module_path = compile_result.module.unwrap_single();
            let bytes = &mut std::fs::File::open(module_path).unwrap();
            let spirv = ash::util::read_spv(bytes)
                .context("Shader crate should be compiled to SPIR-V.")?;

            Ok(ShaderBinary::from(Cow::Owned(spirv.into())))
        }

        let source = handle_compile_result(initial_result)?;

        source_cache.insert(crate_path, source.clone().into());

        Ok(source)
    }

    fn full_path_to_crate(&self, info: &ShaderCompileInfo) -> Result<PathBuf> {
        let ShaderSource::Crate(path) = &info.source else {
            bail!("ShaderCompileInfo must have source type ShaderSource::Crate.")
        };

        if path.is_dir() {
            return Ok(path.clone())
        }

        let full_path = info.compile_options.root_paths.iter().find_map(|root| {
            let shader_crate = [root, &path]
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
                let shader_crate = [root, &path]
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