use daxa_rs::{
    context::*,
    device::*,
    util::pipeline_manager::*,
};

use std::path::PathBuf;

const APPNAME: &str = "Daxa API Test: PipelineManager";
const APPNAME_PREFIX: &str = "[Daxa API Test: PipelineManager]";



#[test]
fn simplest() {
    let context = Context::new(ContextInfo {
        application_name: format!("{} (simplest)", APPNAME).into(),
        application_version: 1,
        ..Default::default()
    }).unwrap();

    let device = context.create_device(DeviceInfo {
        debug_name: format!("{} device (simplest)", APPNAME_PREFIX).into(),
        ..Default::default()
    }).unwrap();


    let mut pipeline_manager = PipelineManager::new(PipelineManagerInfo {
        device: device.clone(),
        shader_compile_options: ShaderCompileOptions {
            root_paths: vec![
                ["tests", "6_pipeline_manager"].iter().collect::<PathBuf>()
            ],
            ..Default::default()
        },
        debug_name: format!("{} pipeline_manager", APPNAME_PREFIX).into()
    }).unwrap();

    let compile_result = pipeline_manager.add_compute_pipeline(ComputePipelineCompileInfo {
        shader_info: ShaderCompileInfo {
            source: ShaderCrate(PathBuf::from("shaders")).into(),
            compile_options: ShaderCompileOptions {
                entry_point: "main_cs",
                ..Default::default()
            }
        },
        push_constant_index: 0,
        debug_name: format!("{} compute_pipeline", APPNAME_PREFIX).into()
    }).unwrap();
}