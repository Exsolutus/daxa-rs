mod task_graph;
use task_graph::AppContext;

use daxa_rs::util::task_graph::*;


const APPNAME: &str = "Daxa API Test: TaskGraph";
const APPNAME_PREFIX: &str = "[Daxa API Test: TaskGraph]";



#[test]
fn simplest() {
    let app = AppContext::new();
    let task_graph = TaskGraph::new(app.device.clone(), TaskGraphInfo {
        debug_name: format!("{} task_graph (simplest)", APPNAME_PREFIX).into(),
        ..Default::default()
    });
}

#[test]
fn execution() {
    let app = AppContext::new();
    let mut task_graph = TaskGraph::new(app.device.clone(), TaskGraphInfo {
        debug_name: format!("{} task_graph (execution)", APPNAME_PREFIX).into(),
        ..Default::default()
    }).unwrap();

    // These are pointless tasks, but demonstrate how the TaskGraph executes
    task_graph.add_task_inline(InlineTaskInfo {
        task: |interface| {
            println!("Hello, ");
        },
        ..Default::default()
    });
    task_graph.add_task_inline(InlineTaskInfo {
        task: |interface| {
            println!("World!");
        },
        ..Default::default()
    });

    task_graph.complete(&TaskCompleteInfo::default());

    task_graph.execute(&ExecutionInfo::default());
}

#[test]
fn write_read_image() {
    // TEST:
    //  1) CREATE image
    //  2) WRITE image
    //  3) READ image
    let app = AppContext::new();
    // Need to scope the task graph lifetime.
    // Task graph MUST drop before we call wait_idle and collect_garbage.
    let mut task_graph = TaskGraph::new(app.device.clone(), TaskGraphInfo {
        debug_name: format!("{} create-write-read image", APPNAME_PREFIX).into(),
        record_debug_information: true,
        ..Default::default()
    }).unwrap();
    // CREATE image
    let task_image = task_graph.create_transient_image(TaskTransientImageInfo {
        size: daxa_rs::types::Extent3D { width: 1, height: 1, depth: 1 },
        name: "Task Graph tested image".into(),
        ..Default::default()
    });
    // WRITE image
    task_graph.add_task_inline(InlineTaskInfo {
        uses: (
            [].into(), 
            [
                TaskImageUse::new(
                    task_image, 
                    TaskImageAccess::ComputeShaderWrite, 
                    ash::vk::ImageViewType::TYPE_2D
                )
            ].into()
        ),
        ..Default::default()
    });
    // READ image
    task_graph.add_task_inline(InlineTaskInfo {
        uses: (
            [].into(),
            [
                TaskImageUse::new(
                    task_image,
                    TaskImageAccess::ComputeShaderRead,
                    ash::vk::ImageViewType::TYPE_2D
                )
            ].into()
        ),
        ..Default::default()
    });

    task_graph.complete(&TaskCompleteInfo::default());
    task_graph.execute(&ExecutionInfo::default());
    println!("{}", task_graph.get_debug_string());
}

#[test]
fn write_read_image_layer() {
    // TEST:
    //  1) CREATE image
    //  2) WRITE into array layer 1 of the image
    //  3) READ from array layer 2 of the image
    let app = AppContext::new();
    // Need to scope the task graph lifetime.
    // Task graph MUST drop before we call wait_idle and collect_garbage.
    let mut task_graph = TaskGraph::new(app.device.clone(), TaskGraphInfo {
        debug_name: format!("{} create-write-read image array layer", APPNAME_PREFIX).into(),
        record_debug_information: true,
        ..Default::default()
    }).unwrap();
    // CREATE image
    let task_image = task_graph.create_transient_image(TaskTransientImageInfo {
        size: daxa_rs::types::Extent3D { width: 1, height: 1, depth: 1 },
        array_layer_count: 2,
        name: "Task Graph tested image".into(),
        ..Default::default()
    });
    // WRITE into array layer 1 of the image
    task_graph.add_task_inline(InlineTaskInfo {
        uses: (
            [].into(), 
            [
                TaskImageUse::new(
                    task_image.view(daxa_rs::types::ImageSubresourceRange {
                        aspect_mask: daxa_rs::gpu_resources::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                        base_mip_level: 0,
                        level_count: 1
                    }), 
                    TaskImageAccess::ComputeShaderWrite, 
                    ash::vk::ImageViewType::TYPE_2D
                )
            ].into()
        ),
        ..Default::default()
    });
    // READ from array layer 2 of the image
    task_graph.add_task_inline(InlineTaskInfo {
        uses: (
            [].into(),
            [
                TaskImageUse::new(
                    task_image.view(daxa_rs::types::ImageSubresourceRange {
                        aspect_mask: daxa_rs::gpu_resources::ImageAspectFlags::COLOR,
                        base_array_layer: 1,
                        layer_count: 1,
                        base_mip_level: 0,
                        level_count: 1
                    }), 
                    TaskImageAccess::ComputeShaderRead, 
                    ash::vk::ImageViewType::TYPE_2D
                )
            ].into()
        ),
        ..Default::default()
    });

    task_graph.complete(&TaskCompleteInfo::default());
    task_graph.execute(&ExecutionInfo::default());
    println!("{}", task_graph.get_debug_string());
}