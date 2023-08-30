mod types;

pub use types::*;
use internal::*;

use crate::{
    command_list::{
        CommandList,
        CommandLabelInfo, 
        ResetSplitBarrierInfo, self, CommandListInfo
    },
    device::{
        Device, PresentInfo, 
    },
    gpu_resources::{
        ImageId, 
    },
    memory_block::MemoryBlock,
    swapchain::{Swapchain, self},
    util::mem::*, 
    types::Access, 
    semaphore::{
        BinarySemaphore, 
        TimelineSemaphore
    }, 
    split_barrier::{
        SplitBarrierWaitInfo, 
        MemoryBarrierInfo,
        ImageBarrierInfo, 
        SplitBarrierSignalInfo
    }, core::Set
};

use anyhow::Result;
use ash::vk::{
    ImageSubresourceRange, 
    ImageLayout
};
use std::{
    borrow::Cow, 
    collections::HashMap,
    sync::atomic::AtomicU32,
};



pub struct TaskInterfaceUses<'a> {
    backend: &'a TaskRuntimeInterface<'a>
}

impl<'a> std::ops::Index<TaskBufferHandle> for TaskInterfaceUses<'a> {
    type Output = &'a TaskBufferUse;

    fn index(&self, index: TaskBufferHandle) -> &Self::Output {
        todo!()
    }
}

impl<'a> std::ops::Index<TaskImageHandle> for TaskInterfaceUses<'a> {
    type Output = &'a TaskImageUse;

    fn index(&self, index: TaskImageHandle) -> &Self::Output {
        todo!()
    }
}

impl<'a> TaskInterfaceUses<'a> {
    fn get_constant_buffer_info(&self) -> crate::command_list::ConstantBufferInfo {
        todo!()
    }
}

pub struct TaskInterface<'a> {
    uses: TaskInterfaceUses<'a>,

    backend: &'a TaskRuntimeInterface<'a>
}

impl<'a> TaskInterface<'a> {
    pub fn get_device(&self) -> &Device {
        todo!()
    }

    pub fn get_command_list(&self) -> &mut CommandList {
        todo!()
    }

    pub fn get_allocator(&self) -> &TransferMemoryPool {
        todo!()
    }
}

pub type TaskCallback = fn(&TaskInterface);

pub struct TaskTransientBufferInfo {
    pub size: u32,
    pub name: String
}

#[derive(Clone)]
pub struct TaskTransientImageInfo {
    pub dimensions: u32,
    pub format: ash::vk::Format,
    pub aspect: crate::gpu_resources::ImageAspectFlags,
    pub size: ash::vk::Extent3D,
    pub mip_level_count: u32,
    pub array_layer_count: u32,
    pub sample_count: u32,
    pub name: String
}

impl Default for TaskTransientImageInfo {
    fn default() -> Self {
        Self {
            dimensions: 2,
            format: ash::vk::Format::R8G8B8A8_UNORM,
            aspect: crate::gpu_resources::ImageAspectFlags::COLOR,
            size: ash::vk::Extent3D {width: 0, height: 0, depth: 0},
            mip_level_count: 1,
            array_layer_count: 1,
            sample_count: 1,
            name: "".into()
        }
    }
}

pub struct TaskBufferAliasInfo {
    pub alias: String,
    pub aliased_buffer: TaskBufferHandle
}

pub struct TaskImageAliasInfo {
    pub alias: String,
    pub aliased_image: TaskImageHandle,
    pub base_mip_level_offset: u32,
    pub base_array_layer_offset: u32
}

pub struct TaskInfo<TaskArgs> {
    pub args: TaskArgs,
    pub task: TaskCallback,
    pub name: String
}

pub struct TaskGraphInfo {
    /// Optionally the user can provide a swapchain. This enables the use of present.
    pub swapchain: Option<Swapchain>,
    /// Task reordering can drastically improve performance, 
    /// yet it is also nice to have sequential callback execution.
    pub reorder_tasks: bool,
    /// Allows TaskGraph to alias transient resource memory (only when that won't break the program)
    pub alias_transients: bool,
    /// Some drivers have bad implementations for split barriers.
    /// In such a case, all use of split barriers can be turned off.
    /// Daxa will use pipeline barriers instead when this is set to false.
    pub use_split_barriers: bool,
    /// Each condition doubled the number of permutations.
    /// 
    /// For a low number of permutations, it is preferable to precompile all permutations.
    /// 
    /// For a high number of permutations, it might be preferable to only compile permutations
    /// just before they are needed. 
    /// 
    /// This JIT (Just In Time) compilations will be enabled when this is set to true.
    pub jit_compile_permutations: bool,
    /// TaskGraph can branch the execution based on conditionals. 
    /// All conditionals must be set before execution and stay constant while executing.
    /// 
    /// This is useful to create permutations of a TaskGraph without having to create a seperate TaskGraph.
    /// Additionally, TaskGraph can generate sync points between executions of permutations,
    /// which cannot be generated between two seperate TaskGraph.
    pub permutation_condition_count: usize,
    /// TaskGraph will put performance markers for use by profilers like NSight around each task execution by default.
    pub enable_command_lables: bool,
    pub task_graph_label_color: [f32; 4],
    pub task_batch_label_color: [f32; 4],
    pub task_label_color: [f32; 4],
    /// Debug information about the execution is recorded when this is set to true.
    /// The result is retrievable with the method [get_debug_string]
    pub record_debug_information: bool,
    /// Sets the size of the linear allocator of device local, host visible memory used by the linear staging allocator.
    /// This memory is used internally as well as by tasks via the [`TaskInterface`]::get_allocator() method.
    /// Setting the size to 0 disables a few TaskGraph features, but also eliminates the memory allocation.
    pub staging_memory_pool_size: u32,
    pub debug_name: Cow<'static, str>
}

impl Default for TaskGraphInfo {
    fn default() -> Self {
        Self {
            swapchain: None,
            reorder_tasks: true,
            alias_transients: false,
            use_split_barriers: true,
            jit_compile_permutations: false,
            permutation_condition_count: 0,
            enable_command_lables: true,
            task_graph_label_color: [0.463, 0.333, 0.671, 1.0],
            task_batch_label_color: [0.563, 0.433, 0.771, 1.0],
            task_label_color: [0.663, 0.533, 0.871, 1.0],
            record_debug_information: false,
            staging_memory_pool_size: 262144,
            debug_name: "".into()
        }
    }
}

#[derive(Default)]
pub struct TaskSubmitInfo {
    pub additional_src_stages: crate::pipeline::PipelineStageFlags,
    pub additional_command_lists: Vec<CommandList>,
    pub additional_wait_binary_semaphores: Vec<BinarySemaphore>,
    pub additional_signal_binary_semaphores: Vec<BinarySemaphore>,
    pub additional_wait_timeline_semaphores: Vec<(TimelineSemaphore, u64)>,
    pub additional_signal_timeline_semaphores: Vec<(TimelineSemaphore, u64)>
}

pub struct TaskPresentInfo {
    pub additional_binary_semaphores: Vec<BinarySemaphore>
}

#[derive(Default)]
pub struct TaskCompleteInfo {

}

pub struct TaskImageLastUse {
    pub range: ImageSubresourceRange,
    pub layout: ImageLayout,
    pub access: Access
}

pub struct TaskGraphConditionalInfo {
    pub condition_index: u32,
    // pub when_true: fn(),
    // pub when_false: fn()
}

#[derive(Default)]
pub struct ExecutionInfo {
    pub permutation_condition_values: Box<[bool]>,
    pub record_debug_string: bool
}

pub struct TrackedBuffers {
    pub buffers: Box<[crate::gpu_resources::BufferId]>,
    pub latest_access: crate::types::Access
}

pub struct TaskBufferInfo {
    pub initial_buffers: TrackedBuffers,
    pub name: String
}

pub struct TaskBuffer {

}

impl TaskBuffer {
    pub fn handle(&self) -> TaskBufferHandle {
        todo!()
    }

    pub fn info(&self) -> &TaskBufferInfo {
        todo!()
    }

    pub fn get_state(&self) -> TrackedBuffers {
        todo!()
    }

    pub fn set_buffers(&mut self, buffers: &TrackedBuffers) {
        todo!()
    }

    pub fn swap_buffers(&mut self, other: &mut TaskBuffer) {
        todo!()
    }
}

pub struct TrackedImages {
    pub images: Box<[ImageId]>,
    pub latest_range_states: Box<[ImageRangeState]>
}

pub struct TaskImageInfo {
    pub initial_images: TrackedImages,
    pub swapchain_image: bool,
    pub name: String
}

#[derive(Default)]
pub struct TaskImage {

}

impl TaskImage {
    pub fn new(info: TaskImageInfo) -> Self {
        todo!()
    }
}

impl TaskImage {
    pub fn handle(&self) -> TaskImageHandle {
        todo!()
    }

    pub fn info(&self) -> &TaskImageInfo {
        todo!()
    }

    pub fn get_state(&self) -> TrackedImages {
        todo!()
    }

    pub fn set_images(&mut self, images: &TrackedImages) {
        todo!()
    }

    pub fn swap_images(&mut self, other: &mut TaskImage) {
        todo!()
    }
}

pub struct InlineTaskInfo {
    pub uses: (Box<[TaskBufferUse]>, Box<[TaskImageUse]>),
    pub task: TaskCallback,
    pub constant_buffer_slot: isize,
    pub name: Cow<'static, str>
}

impl Default for InlineTaskInfo {
    fn default() -> Self {
        Self {
            uses: Default::default(),
            task: |_| { },
            constant_buffer_slot: -1,
            name: "".into()
        }
    }
}



pub struct TaskGraph {
    exec_unique_next_index: AtomicU32,
    unique_index: u32,

    preamble: TaskCallback,
    device: Device,
    info: TaskGraphInfo,
    global_buffer_infos: Vec<PermIndepTaskBufferInfo>,
    global_image_infos: Vec<PermIndepTaskImageInfo>,
    permutations: Vec<TaskGraphPermutation>,
    tasks: Vec<Task>,
    persistent_buffer_index_to_local_index: HashMap<u32, u32>,
    persistent_image_index_to_local_index: HashMap<u32, u32>,

    // record time information
    record_active_conditional_scopes: u32,
    record_conditional_states: u32,
    record_active_permutations: Vec<u32>,
    buffer_name_to_id: HashMap<String, TaskBufferHandle>,
    image_name_to_id: HashMap<String, TaskImageHandle>,

    memory_block_size: u64,
    memory_type_bits: u32,
    transient_data_memory_block: Option<MemoryBlock>,
    compiled: bool,

    // execution time information
    staging_memory: TransferMemoryPool,
    execution_time_current_conditionals: [bool; DAXA_TASKGRAPH_MAX_CONDITIONALS],

    // post execution information
    last_execution_staging_timeline_value: usize,
    chosen_permutation_last_execution: u32,
    left_over_command_lists: Vec<CommandList>,
    executed_once: bool,
    prev_frame_permutation_index: u32,
    debug_string_stream: String
}


// TaskGraph creation methods
impl TaskGraph {
    pub fn new(device: Device, info: TaskGraphInfo) -> Result<Self> {
        let staging_memory = TransferMemoryPool::new(device.clone(), TransferMemoryPoolInfo {
            capacity: info.staging_memory_pool_size,
            debug_name: info.debug_name.clone()
        })?;
        let mut permutations = Vec::<TaskGraphPermutation>::new();
        permutations.resize_with(1usize << &info.permutation_condition_count, || TaskGraphPermutation::default() );

        let mut task_graph = TaskGraph {
            preamble: |_| { },
            device,
            info,
            exec_unique_next_index: 1.into(),
            unique_index: 0,
            global_buffer_infos: Default::default(),
            global_image_infos: Default::default(),
            permutations,
            tasks: Default::default(),
            persistent_buffer_index_to_local_index: Default::default(),
            persistent_image_index_to_local_index: Default::default(),

            record_active_conditional_scopes: 0,
            record_conditional_states: 0,
            record_active_permutations: Default::default(),
            buffer_name_to_id: Default::default(),
            image_name_to_id: Default::default(),

            memory_block_size: 0,
            memory_type_bits: 0xFFFFFFFFu32,
            transient_data_memory_block: None,
            compiled: false,

            staging_memory,
            execution_time_current_conditionals: Default::default(),

            last_execution_staging_timeline_value: 0,
            chosen_permutation_last_execution: 0,
            left_over_command_lists: Default::default(),
            executed_once: false,
            prev_frame_permutation_index: 0,
            debug_string_stream: "".into()
        };

        for permutation in &mut task_graph.permutations {
            permutation.batch_submit_scopes.push(TaskBatchSubmitScope::default());
        }
        task_graph.update_active_permutations();

        Ok(task_graph)
    }
}

impl Drop for TaskGraph {
    fn drop(&mut self) {
        let mut permutations = std::mem::take(&mut self.permutations);
        for permutation in &mut permutations {
            // Because transient buffers are owned by TaskGraph, we need to destroy them here
            for buffer_info_index in 0..self.global_buffer_infos.len() {
                let global_buffer = &self.global_buffer_infos[buffer_info_index];
                let permutation_buffer = &mut permutation.buffer_infos[buffer_info_index];
                if let PermIndepTaskBufferInfo::Transient { .. } = global_buffer {
                    if permutation_buffer.valid {
                        self.device.destroy_buffer(self.get_actual_buffers(
                            TaskBufferHandle::Transient { task_graph_index: self.unique_index, index: buffer_info_index as u32 },
                            &permutation
                        )[0]);
                    }
                }
            }
            // Because transient images are owned by TaskGraph, we need to destroy them here
            for image_info_index in 0..self.global_image_infos.len() {
                let global_image = &self.global_image_infos[image_info_index];
                let permutation_image = &mut permutation.image_infos[image_info_index];
                if let PermIndepTaskImageInfo::Transient { info, memory_requirements } = global_image {
                    if permutation_image.valid {
                        self.device.destroy_image(self.get_actual_images(
                            TaskImageHandle::Transient {
                                task_graph_index: self.unique_index,
                                index: image_info_index as u32,
                                range: Default::default()
                            },
                            permutation
                        )[0]);
                    }
                }
            }
        }
        for task in &self.tasks {
            for view_cache in &task.image_view_cache {
                for &view in view_cache {
                    let parent = self.device.info_image_view(view).image;
                    let is_default_view = parent.default_view() == view;
                    if !is_default_view {
                        self.device.destroy_image_view(view);
                    }
                }
            }
        }
    }
}

// TaskGraph usage methods
impl TaskGraph {
    pub fn use_persistent_buffer(&mut self, buffer: TaskBuffer) {
        todo!()
    }

    pub fn use_persistent_image(&mut self, image: TaskImage) {
        todo!()
    }

    pub fn create_transient_buffer(&mut self, info: TaskTransientBufferInfo) -> TaskBufferHandle {
        todo!()
    }

    pub fn create_transient_image(&mut self, info: TaskTransientImageInfo) -> TaskImageHandle {
        debug_assert!(!self.compiled, "Can't create resources on a completed task graph!");
        debug_assert!(!self.image_name_to_id.contains_key(&info.name), "Task image names must be unique!");

        let task_image_id = TaskImageHandle::Transient {
            task_graph_index: self.unique_index,
            index: self.global_image_infos.len() as u32,
            range: ImageSubresourceRange {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            }
        };

        for permutation in self.permutations.iter_mut() {
            permutation.image_infos.push(PerPermTaskImage {
                valid: permutation.active,
                swapchain_semaphore_waited_upon: false,
                ..Default::default()
            })
        }

        self.global_image_infos.push(PermIndepTaskImageInfo::Transient {
            info: info.clone(),
            memory_requirements: Default::default()
        });
        self.image_name_to_id.insert(info.name, task_image_id);

        task_image_id
    }

    pub fn add_task<T>(&mut self, task: T) {
        todo!()
    }

    pub fn add_task_inline(&mut self, info: InlineTaskInfo) {
        if self.compiled {
            #[cfg(debug_assertions)]
            panic!("Can't record tasks on a completed task graph!");
            #[cfg(not(debug_assertions))]
            unreachable!();
        }

        self.add_task_internal(InlineTask {
            uses: info.uses,
            callback_lambda: info.task,
            name: info.name.into(),
            constant_buffer_slot: info.constant_buffer_slot
        });
    }

    // pub fn conditional(&self, info: TaskConditionalInfo) {
    //     todo!()
    // }

    // pub fn submit(&self, info: TaskSubmitInfo) {
    //     todo!()
    // }

    // pub fn present(&self, info: TaskPresentInfo) {
    //     todo!()
    // }

    pub fn complete(&mut self, info: &TaskCompleteInfo) {
        debug_assert!(!self.compiled, "TaskGraph can only be completed once!");
        self.compiled = true;

        if self.global_buffer_infos.is_empty() && self.global_image_infos.is_empty() {
            return;
        }

        self.allocate_transient_resources();
        // Insert static barriers initializing image layouts.
        let mut permutations = std::mem::take(&mut self.permutations);
        for permutation in permutations.iter_mut() {
            self.create_transient_runtime_buffers(permutation);
            self.create_transient_runtime_images(permutation);
            // Insert static initialization barriers for non persistent resources:
            // Buffers never need layout initialization, only images.
            for task_image_index in 0..permutation.image_infos.len() {
                let task_image_handle = TaskImageHandle::Transient {
                    task_graph_index: self.unique_index,
                    index: task_image_index as u32,
                    range: Default::default()
                };
                let task_image = &permutation.image_infos[task_image_index];
                let global_task_image = &self.global_image_infos[task_image_index];
                if task_image.valid {
                    if let PermIndepTaskImageInfo::Transient { .. } = global_task_image {
                        // Insert barriers, initializing all the initially accessed subresource ranges to the correct layout.
                        for first_access in &task_image.first_range_states {
                            let new_barrier_index = permutation.barriers.len();
                            permutation.barriers.push(TaskBarrier {
                                image_id: Some(task_image_handle),
                                range: first_access.state.range,
                                src_layout: Default::default(),
                                dst_layout: first_access.state.latest_layout,
                                src_access: Default::default(),
                                dst_access: first_access.state.latest_access,
                            });
                            // Because resources may be aliased, we need to insert the barrier into the batch where the resource is first used.
                            // If we jut inserted all initialization transitions into the first batch, and error might occur as follows:
                            //      Image A lives in Batch 1, Image B lives in Batch 2
                            //      Image A and B are aliased (share the same memory, either fully or partially)
                            //      Image A is transitioned from UNDEFINED -> TRANSFER_DST in Batch 0, BUT
                            //      Image B is also transitioned from UNDEFINED -> TRANSFER_SRC in Batch 0
                            // This is an erroneous state - TaskGraph assumes they are seperate images, and thus
                            // uses Image A thinking it is in TRANSFER_DST while it is not
                            if self.info.alias_transients {
                                // TODO(msakmary) This is only needed when we actually alias two images - should be possible to detect this
                                // and only defer the initialization barrier for these aliased ones instead of all of them
                                let submit_scope_index = first_access.latest_access_submit_scope_index;
                                let batch_index = first_access.latest_access_batch_index;
                                let first_used_batch = &mut permutation.batch_submit_scopes[submit_scope_index].task_batches[batch_index];
                                first_used_batch.pipeline_barrier_indices.push(new_barrier_index);
                            } else {
                                let first_used_batch = &mut permutation.batch_submit_scopes[0].task_batches[0];
                                first_used_batch.pipeline_barrier_indices.push(new_barrier_index);
                            }
                        }
                    }
                }
            }
        }
        // Return temporarily owned fields
        self.permutations = permutations;
    }

    /// Execution flow:
    /// 1. choose permutation based on conditionals
    /// 2. validate used persistent resources, based on permutation
    /// 3. runtime generate and insert runtime sync for persistent resources.
    /// 4. for every submit scope:
    ///     2.1 for every batch in scope:
    ///         3.1 wait for pipeline and split barriers
    ///         3.2 for every task:
    ///             4.1 validate runtime resources of used resources
    ///             4.2 refresh image view cache.
    ///             4.3 collect shader use handles, allocate gpu local staging memory, copy in handles and bind to constant buffer binding.
    ///             4.4 run task
    ///         3.3 signal split barriers
    ///     2.2 check if submit scope submits work, either submit or collect cmd lists and sync primitives for querry
    ///     2.3 check if submit scope presents, present if true.
    pub fn execute(&mut self, info: &ExecutionInfo) {
        debug_assert!(info.permutation_condition_values.len() >= self.info.permutation_condition_count, "Detected invalid permutation condition count!");
        debug_assert!(self.compiled, "TaskGraph must be completed before execution!");

        let mut permutation_index = 0;
        for index in 0..info.permutation_condition_values.len().min(32) {
            permutation_index |= match info.permutation_condition_values[index] {
                true => 1 << index,
                false => 0
            };
        }
        self.chosen_permutation_last_execution = permutation_index as u32;
        // Temporarily extract permutations for mutability
        let mut permutations = std::mem::take(&mut self.permutations);
        let permutation = &mut permutations[permutation_index];

        // Temporarily extract batch_submit_scopes for mutability
        let mut batch_submit_scopes = std::mem::take(&mut permutation.batch_submit_scopes);

        let mut runtime = TaskRuntimeInterface {
            //task_graph: Some(&self),
            permutation: Some(permutation),
            ..Default::default()
        };
        runtime.command_lists.push(self.device.create_command_list(crate::command_list::CommandListInfo {
            debug_name: format!("Task CommandList {}", runtime.command_lists.len()).into()
        }).expect("CommandList should be created."));

        self.validate_runtime_resources(&permutation);
        // Generate and insert synchronization for persistent resources:
        self.generate_persistent_resource_sync(&permutation, runtime.command_lists.last_mut().unwrap());
        
        let submit_scope_count = batch_submit_scopes.len();
        for (submit_scope_index, submit_scope) in batch_submit_scopes.iter_mut().enumerate() {
            if self.info.enable_command_lables {
                runtime.command_lists.last_mut().unwrap().begin_label(CommandLabelInfo {
                    label_name: format!("{}, submit {}", self.info.debug_name, submit_scope_index).into(),
                    label_color: self.info.task_batch_label_color
                })
            }

            for (batch_index, task_batch) in submit_scope.task_batches.iter().enumerate() {
                if self.info.enable_command_lables {
                    runtime.command_lists.last_mut().unwrap().begin_label(CommandLabelInfo {
                        label_name: format!("{}, submit {}, batch {}", self.info.debug_name, submit_scope_index, batch_index).into(),
                        label_color: self.info.task_batch_label_color
                    });
                }
                // Wait on pipeline barriers before batch execution.
                for &barrier_index in &task_batch.pipeline_barrier_indices {
                    let barrier = &permutation.barriers[barrier_index];
                    self.insert_pipeline_barrier(permutation, runtime.command_lists.last_mut().unwrap(), barrier);
                }
                // Wait on split barriers before batch execution.
                let mut needed_image_barriers = 0;
                if !self.info.use_split_barriers {
                    for &barrier_index in &task_batch.wait_split_barrier_indices {
                        let split_barrier = &permutation.split_barriers[barrier_index];
                        // Convert split barrier to normal barrier.
                        let barrier = TaskBarrier {
                            image_id: split_barrier.image_id,
                            range: split_barrier.range,
                            src_layout: split_barrier.src_layout,
                            dst_layout: split_barrier.dst_layout,
                            src_access: split_barrier.src_access,
                            dst_access: split_barrier.dst_access,
                        };
                        self.insert_pipeline_barrier(permutation, runtime.command_lists.last_mut().unwrap(), &barrier);
                    }
                } else {
                    for &barrier_index in &task_batch.wait_split_barrier_indices {
                        let split_barrier = &permutation.split_barriers[barrier_index];
                        if let Some(image_id) = split_barrier.image_id {
                            needed_image_barriers += self.get_actual_images(image_id, permutation).len();
                        }
                    }
                    let mut split_barrier_wait_infos = Vec::<SplitBarrierWaitInfo>::with_capacity(task_batch.wait_split_barrier_indices.len());
                    let mut memory_barrier_infos = Vec::<MemoryBarrierInfo>::with_capacity(task_batch.wait_split_barrier_indices.len());
                    let mut image_barrier_infos = Vec::<ImageBarrierInfo>::with_capacity(needed_image_barriers);
                    for &barrier_index in &task_batch.wait_split_barrier_indices {
                        let split_barrier = &permutation.split_barriers[barrier_index];
                        if let Some(image_id) = split_barrier.image_id {
                            let image_barrier_vec_start_size = image_barrier_infos.len();
                            for &image in self.get_actual_images(image_id, permutation).iter() {
                                image_barrier_infos.push(ImageBarrierInfo {
                                    src_access: split_barrier.src_access,
                                    dst_access: split_barrier.dst_access,
                                    src_layout: split_barrier.src_layout,
                                    dst_layout: split_barrier.dst_layout,
                                    range: split_barrier.range,
                                    image
                                });
                            }
                            let image_barrier_vec_end_size = image_barrier_infos.len();
                            split_barrier_wait_infos.push(SplitBarrierWaitInfo {
                                memory_barriers: Default::default(),
                                image_barriers: unsafe { Box::from_raw(&mut image_barrier_infos[image_barrier_vec_start_size..image_barrier_vec_end_size]) },
                                split_barrier: split_barrier.split_barrier_state.clone()
                            })
                        } else {
                            let barrier_info = MemoryBarrierInfo {
                                src_access: split_barrier.src_access,
                                dst_access: split_barrier.dst_access
                            };
                            memory_barrier_infos.push(barrier_info.clone());
                            split_barrier_wait_infos.push(SplitBarrierWaitInfo {
                                memory_barriers: unsafe { Box::from_raw(&mut [barrier_info]) },
                                image_barriers: Default::default(),
                                split_barrier: split_barrier.split_barrier_state.clone()
                            })
                        }
                    }
                    if !split_barrier_wait_infos.is_empty() {
                        runtime.command_lists.last_mut().unwrap().wait_split_barriers(&split_barrier_wait_infos);
                    }
                }
                // Execute all tasks in the batch.
                for (task_index, &task_id) in task_batch.tasks.iter().enumerate() {
                    self.execute_task(&mut runtime, permutation, task_index, task_id)
                }
                if self.info.use_split_barriers {
                    // Reset all waited split barriers here.
                    for &barrier_index in &task_batch.wait_split_barrier_indices {
                        // We wait on the stages that waited on our split barrier earlier.
                        // This way we make sure that the stages that wait on the split barrier
                        // executed and saw the split barrier signaled before we reset them.
                        runtime.command_lists.last_mut().unwrap().reset_split_barrier(ResetSplitBarrierInfo {
                            barrier: permutation.split_barriers[barrier_index].split_barrier_state.clone(),
                            stage: permutation.split_barriers[barrier_index].dst_access.0
                        });
                    }
                    // Signal all signal split barriers after batch execution.
                    for &barrier_index in &task_batch.signal_split_barrier_indices {
                        let task_split_barrier = &permutation.split_barriers[barrier_index];
                        if let Some(image_id) = task_split_barrier.image_id {
                            let mut image_barrier_infos = Vec::<ImageBarrierInfo>::with_capacity(needed_image_barriers);
                            for &image in self.get_actual_images(image_id, permutation).iter() {
                                image_barrier_infos.push(ImageBarrierInfo {
                                    src_access: task_split_barrier.src_access,
                                    dst_access: task_split_barrier.dst_access,
                                    src_layout: task_split_barrier.src_layout,
                                    dst_layout: task_split_barrier.dst_layout,
                                    range: task_split_barrier.range,
                                    image
                                });
                            }
                            runtime.command_lists.last_mut().unwrap().signal_split_barrier(SplitBarrierSignalInfo {
                                memory_barriers: Default::default(),
                                image_barriers: unsafe { Box::from_raw(image_barrier_infos.as_mut_slice()) },
                                split_barrier: task_split_barrier.split_barrier_state.clone()
                            });
                        } else {
                            let memory_barrier = MemoryBarrierInfo {
                                src_access: task_split_barrier.src_access,
                                dst_access: task_split_barrier.dst_access
                            };
                            runtime.command_lists.last_mut().unwrap().signal_split_barrier(SplitBarrierSignalInfo {
                                memory_barriers: unsafe { Box::from_raw(&mut [memory_barrier]) },
                                image_barriers: Default::default(),
                                split_barrier: task_split_barrier.split_barrier_state.clone()
                            })
                        }
                    }
                }
                if self.info.enable_command_lables {
                    runtime.command_lists.last_mut().unwrap().end_label()
                }
            }
            for &barrier_index in &submit_scope.last_minute_barrier_indices {
                let barrier = &permutation.barriers[barrier_index];
                self.insert_pipeline_barrier(permutation, runtime.command_lists.last_mut().unwrap(), barrier);
            }
            if self.info.enable_command_lables {
                runtime.command_lists.last_mut().unwrap().end_label();
            }
            runtime.command_lists = runtime.command_lists
                .drain(..)
                .map(|command_list| {
                    debug_assert!(!command_list.is_complete(), "It is illegal to complete command lists in tasks that are obtained by the runtime!");
                    command_list.complete()
                })
                .collect();

            if submit_scope_index != submit_scope_count - 1 {
                let submit_info = &mut submit_scope.submit_info;
                submit_info.command_lists.append(&mut runtime.command_lists);
                if let Some(swapchain) = &self.info.swapchain {
                    if submit_scope_index == permutation.swapchain_image_first_use_submit_scope_index {
                        submit_info.wait_binary_semaphores.push(swapchain.get_acquire_semaphore());
                    } else if submit_scope_index == permutation.swapchain_image_last_use_submit_scope_index {
                        submit_info.signal_binary_semaphores.push(swapchain.get_present_semaphore());
                        submit_info.signal_timeline_semaphores.push((
                            swapchain.get_gpu_timeline_semaphore(), 
                            swapchain.get_cpu_timeline_value() as u64
                        ));
                    }
                }
                if !submit_scope.user_submit_info.additional_command_lists.is_empty() {
                    submit_info.command_lists.append(&mut submit_scope.user_submit_info.additional_command_lists);
                }
                if !submit_scope.user_submit_info.additional_wait_binary_semaphores.is_empty() {
                    submit_info.wait_binary_semaphores.append(&mut submit_scope.user_submit_info.additional_wait_binary_semaphores);
                }
                if !submit_scope.user_submit_info.additional_signal_binary_semaphores.is_empty() {
                    submit_info.signal_binary_semaphores.append(&mut submit_scope.user_submit_info.additional_signal_binary_semaphores);
                }
                if !submit_scope.user_submit_info.additional_wait_timeline_semaphores.is_empty() {
                    submit_info.wait_timeline_semaphores.append(&mut submit_scope.user_submit_info.additional_wait_timeline_semaphores);
                }
                if !submit_scope.user_submit_info.additional_signal_timeline_semaphores.is_empty() {
                    submit_info.signal_timeline_semaphores.append(&mut submit_scope.user_submit_info.additional_signal_timeline_semaphores);
                }
                if self.staging_memory.timeline_value() > self.last_execution_staging_timeline_value {
                    submit_info.signal_timeline_semaphores.push((
                        self.staging_memory.get_timeline_semaphore(),
                        self.staging_memory.timeline_value() as u64
                    ));
                }
                self.device.submit_commands(submit_info);

                if let Some((mut present_wait_semaphores, mut additional_present_semaphores)) = submit_scope.present_semaphores.clone() {
                    if let Some(swapchain) = &self.info.swapchain {
                        present_wait_semaphores.push(swapchain.get_present_semaphore());
                        if !additional_present_semaphores.is_empty() {
                            present_wait_semaphores.append(&mut additional_present_semaphores);
                        }
                        self.device.preset_frame(PresentInfo {
                            wait_binary_semaphores: present_wait_semaphores,
                            swapchain
                        });
                    }
                }
                // We need to cleare all completed command lists that have been submitted.
                runtime.command_lists.clear();
                runtime.command_lists.push(self.device.create_command_list(CommandListInfo {
                    debug_name: format!("Task CommandList {}", runtime.command_lists.len()).into()
                }).unwrap());
            }
        }

        self.left_over_command_lists = std::mem::take(&mut runtime.command_lists);
        self.executed_once = true;
        self.prev_frame_permutation_index = permutation_index as u32;
        self.last_execution_staging_timeline_value = self.staging_memory.timeline_value();

        // Insert previous uses into execution info for the next execution's sync.
        for task_buffer_index in 0..permutation.buffer_infos.len() {
            if let PermIndepTaskBufferInfo::Persistent { buffer } = &mut self.global_buffer_infos[task_buffer_index] {
                if permutation.buffer_infos[task_buffer_index].valid {
                    buffer.latest_access = permutation.buffer_infos[task_buffer_index].latest_access;
                }
            }
        }
        for task_image_index in 0..permutation.image_infos.len() {
            if let PermIndepTaskImageInfo::Persistent { image } = &mut self.global_image_infos[task_image_index] {
                if permutation.image_infos[task_image_index].valid {
                    for extended_state in &permutation.image_infos[task_image_index].last_range_states {
                        image.latest_range_states.push(extended_state.state);
                    }
                }
            }
        }

        // Return ownership of extracted fields
        permutation.batch_submit_scopes = batch_submit_scopes;
        self.permutations = permutations;

        if self.info.record_debug_information {
            self.debug_print();
        }
    }

    fn validate_runtime_resources(&self, permutation: &TaskGraphPermutation) {
        #[cfg(debug_assertions)]
        {
            const PERSISTENT_RESOURCE_MESSAGE: &str = "When executing a task graph, all used persistent resources must be backed one or more valid runtime resources.";
            for local_buffer_index in 0..self.global_buffer_infos.len() {
                if !permutation.buffer_infos[local_buffer_index].valid {
                    continue;
                }
                if let PermIndepTaskBufferInfo::Persistent { buffer } = &self.global_buffer_infos[local_buffer_index] {
                    let runtime_buffers = &buffer.actual_buffers;
                    debug_assert!(
                        runtime_buffers.len() > 0, 
                        "Detected persistent task buffer \"{}\" used in task graph \"{}\" with 0 runtime buffers; {}",
                        buffer.info.name,
                        self.info.debug_name,
                        PERSISTENT_RESOURCE_MESSAGE
                    );
                }
                
            }
            for local_image_index in 0..self.global_image_infos.len() {
                if !permutation.image_infos[local_image_index].valid {
                    continue;
                }
                if let PermIndepTaskImageInfo::Persistent { image } = &self.global_image_infos[local_image_index] {
                    let runtime_images = &image.actual_images;
                    debug_assert!(
                        runtime_images.len() > 0,
                        "Detected persistent task image \"{}\" used in task graph \"{}\" with 0 runtime images; {}",
                        image.info.name,
                        self.info.debug_name,
                        PERSISTENT_RESOURCE_MESSAGE
                    );
                }
            }
        }
    }

    fn generate_persistent_resource_sync(&mut self, permutation: &TaskGraphPermutation, command_list: &mut CommandList) {
        // Persistent resources need just-in-time sync between executions as
        // pregenerating the transitions between all permutations is not manageable.
        let mut out = String::new();
        let mut indent = String::new();
        if self.info.record_debug_information {
            out += "Runtime sync memory barriers:\n"
        }
        for task_buffer_index in 0..permutation.buffer_infos.len() {
            let task_buffer = &permutation.buffer_infos[task_buffer_index];
            let global_buffer_info = &mut self.global_buffer_infos[task_buffer_index];
            if task_buffer.valid {
                if let PermIndepTaskBufferInfo::Persistent { buffer } = global_buffer_info {
                    use ash::vk::AccessFlags2;
                    let no_previous_access = buffer.latest_access.1 == AccessFlags2::NONE;
                    let read_on_read_same_access = buffer.latest_access == permutation.buffer_infos[task_buffer_index].first_access
                                                && buffer.latest_access.1 == AccessFlags2::MEMORY_READ;
                    // TODO(pahrens, msakymary): read on read should only be true whenever the accesses are the same AND the stages are also the same,
                    // otherwise we need to generate a barrier
                    //      AS WE CAN'T MODIFY BARRIERS FROM PREVIOUS ALREADY EXECUTED TASK GRAPHS, WE MUST OVERSYNC ON THE LAST WRITE TO READ.
                    //      WE CAN ONLY SKIP ON THE SAME ACCESS READ ON READ
                    //      WE MUST REMEMBER THE LAST WRITE

                    // let read_on_read = buffer.latest_access == permutation.buffer_infos[task_buffer_index].first_access
                    //                     && buffer.latest_access.1 == AccessFlags2::MEMORY_READ;
                    // For now just oversync on reads.
                    if no_previous_access || read_on_read_same_access {
                        // Skip buffers that have no previous access, as there is nothing to sync on.
                        continue;
                    }

                    let memory_barrier_info = MemoryBarrierInfo {
                        src_access: buffer.latest_access,
                        dst_access: permutation.buffer_infos[task_buffer_index].first_access
                    };
                    if self.info.record_debug_information {
                        out += format!("\t{:?}\n", memory_barrier_info).as_str();
                    }
                    command_list.pipeline_barrier(memory_barrier_info);
                    buffer.latest_access = Default::default();
                }
            }
        }
        if self.info.record_debug_information {
            out += "Runtime sync image memory barriers:\n";
        }
        // If parts of the first use range do not intersect with any previous use,
        // we must synchronize on undefined layout!
        let mut remaining_first_accesses = Vec::<ExtendedImageRangeState>::new();
        for task_image_index in 0..permutation.image_infos.len() {
            let task_image = &permutation.image_infos[task_image_index];
            let execution_image = &mut self.global_image_infos[task_image_index];
            remaining_first_accesses = task_image.first_range_states.clone();
            // Iterate over all persistent images.
            // Find all intersections between tracked ranges o first use and previous use.
            // Sync on the intersection and delete the intersected part from the tracked range of the previous use.
            if task_image.valid {
                if let PermIndepTaskImageInfo::Persistent { image } = execution_image {
                    if self.info.record_debug_information {
                        out += "\tSync from previous uses:\n";
                    }
                    let previous_access_ranges = &mut image.latest_range_states;
                    let mut previous_access_range_index = 0;
                    while previous_access_range_index < previous_access_ranges.len() {
                        let mut broke_inner_loop = false;
                        let previous_access_range = previous_access_ranges[previous_access_range_index];
                        let mut first_access_range_index = 0;
                        while first_access_range_index < remaining_first_accesses.len() {
                            let first_access_range = remaining_first_accesses[first_access_range_index];
                            // Don't sync on disjoin subresource uses.
                            if !first_access_range.state.range.intersects(previous_access_range.range) {
                                // Disjoint subresources or read on read with same layout.
                                continue;
                            }
                            // Intersect previous use and initial use.
                            // Record synchronization for the intersecting part.
                            let intersection = previous_access_range.range.intersect(first_access_range.state.range);
                            // Don't sync on same accesses following each other.
                            use ash::vk::AccessFlags2;
                            let both_accesses_read = first_access_range.state.latest_access.1 == AccessFlags2::MEMORY_READ
                                                    && previous_access_range.latest_access.1 == AccessFlags2::MEMORY_READ;
                            let both_layouts_same = first_access_range.state.latest_layout == previous_access_range.latest_layout;
                            if !(both_accesses_read && both_layouts_same) {
                                for image_id in &image.actual_images {
                                    let image_barrier_info = ImageBarrierInfo {
                                        src_access: previous_access_range.latest_access,
                                        dst_access: first_access_range.state.latest_access,
                                        src_layout: previous_access_range.latest_layout,
                                        dst_layout: first_access_range.state.latest_layout,
                                        range: intersection,
                                        image: image_id.clone()
                                    };
                                    if self.info.record_debug_information {
                                        out += format!("\t\t{:?}\n", image_barrier_info).as_str();
                                    }
                                    command_list.pipeline_barrier_image_transition(image_barrier_info);
                                }
                            }
                            // Put back the non intersecting rest into the previous use list.
                            let (previous_use_range_rest, previous_use_range_rest_count) = previous_access_range.range.subtract(intersection);
                            let (first_use_range_rest, first_use_range_rest_count) = first_access_range.state.range.subtract(intersection);
                            for rest_range_index in 0..previous_use_range_rest_count {
                                let mut rest_previous_range = previous_access_range.clone();
                                rest_previous_range.range = previous_use_range_rest[rest_range_index];
                                previous_access_ranges.push(rest_previous_range);
                            }
                            // Append the new rest first uses.
                            for rest_range_index in 0..first_use_range_rest_count {
                                let mut rest_first_range = first_access_range.clone();
                                rest_first_range.state.range = first_use_range_rest[rest_range_index];
                                remaining_first_accesses.push(rest_first_range);
                            }
                            // Remove the previous use from the list, it is synchronized now.
                            previous_access_ranges.remove(previous_access_range_index);
                            // Remove the first use from the remaining first uses, as it was now synchronized from.
                            remaining_first_accesses.remove(first_access_range_index);
                            // As we removed an element from this place,
                            // we don't need to advance the index as in its place 
                            // there will be a new element already that we do not want to skip.
                            broke_inner_loop = true;
                            break;
                        }
                        if !broke_inner_loop {
                            // We break the loop when we remove the current element.
                            // Removing moved all elements past the current one to the left.
                            // This means that the current element is already a new one.
                            // We do not want to skip it so we decrement the index here.
                            previous_access_range_index += 1;
                        }
                    }

                    if self.info.record_debug_information {
                        out += "\tSync from undefined:\n";
                    }
                    // For all first uses that did NOT intersect with a previous use,
                    // we need to syncronize from an undefined state to initialize the layout of the image.
                    for remaining_first_uses_index in 0..remaining_first_accesses.len() {
                        for &image_id in self.get_actual_images(
                            TaskImageHandle::Transient { 
                                task_graph_index: self.unique_index, 
                                index: task_image_index as u32, 
                                range: Default::default() 
                            },
                            permutation
                        ).iter() {
                            let image_barrier_info = ImageBarrierInfo {
                                src_access: crate::types::access_consts::NONE,
                                dst_access: remaining_first_accesses[remaining_first_uses_index].state.latest_access,
                                src_layout: ImageLayout::UNDEFINED,
                                dst_layout: remaining_first_accesses[remaining_first_uses_index].state.latest_layout,
                                range: remaining_first_accesses[remaining_first_uses_index].state.range,
                                image: image_id
                            };
                            if self.info.record_debug_information {
                                out += format!("\t\t{:?}\n", image_barrier_info).as_str();
                            }
                            command_list.pipeline_barrier_image_transition(image_barrier_info);
                        }
                    }
                }
            }
        }

        if self.info.record_debug_information {
            self.debug_string_stream += &out;
        }
    }

    fn insert_pipeline_barrier(&self, permutation: &TaskGraphPermutation, command_list: &mut CommandList, barrier: &TaskBarrier) {
        // Check if barrier is an image barrier or a normal barrier
        if let Some(image_id) = barrier.image_id {
            let actual_images = self.get_actual_images(image_id, permutation);
            for &image in actual_images.iter() {
                command_list.pipeline_barrier_image_transition(ImageBarrierInfo {
                    src_access: barrier.src_access,
                    dst_access: barrier.dst_access,
                    src_layout: barrier.src_layout,
                    dst_layout: barrier.dst_layout,
                    range: barrier.range,
                    image
                });
            }
        } else {
            command_list.pipeline_barrier(MemoryBarrierInfo {
                src_access: barrier.src_access,
                dst_access: barrier.dst_access
            });
        }
    }

    pub fn get_command_lists(&self) -> Vec<CommandList> {
        todo!()
    }

    pub fn get_debug_string(&self) -> String {
        debug_assert!(
            self.info.record_debug_information,
            "In order to have debug string you need to set record_debug_information flag to true on task graph creation."
        );
        debug_assert!(
            self.executed_once,
            "In order to have debug string you need to execute the task graph at least once."
        );

        self.debug_string_stream.clone()
    }
}



mod internal {
    use super::*;
    
    use crate::{
        command_list::CommandList,
        device::{Device, CommandSubmitInfo},
        gpu_resources::{
            BufferId,
            ImageId,
            ImageViewId,
            BufferInfo, ImageInfo
        },
        util::mem::*, 
        types::Access, 
        split_barrier::{
            SplitBarrierState,
            SplitBarrierInfo
        }, 
        semaphore::BinarySemaphore, 
        core::Set, 
        memory_block::MemoryBlockInfo
    };
    
    use anyhow::Result;
    use ash::vk::{
        ImageUsageFlags, 
        ImageSubresourceRange, ImageLayout
    };
    use std::{
        sync::atomic::AtomicU32,
        collections::HashMap,
    };



    pub(super) const DAXA_TASKGRAPH_MAX_CONDITIONALS: usize = 31;

    pub(super) type TaskBatchId = usize;
    pub(super) type TaskId = usize;
    

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(super) enum LastReadIndex {
        None,
        Barrier(usize),
        SplitBarrier(usize)
    }
    
    // TODO(msakmary, pahrens) This will not be needed once we make batches linear and not
    // contained in submit scopes
    pub(super) struct CombinedBatchIndex {
        pub submit_scope_index: usize,
        pub task_batch_index: usize
    }
    
    impl Default for CombinedBatchIndex {
        fn default() -> Self {
            Self {
                submit_scope_index: usize::MAX,
                task_batch_index: usize::MAX
            }
        }
    }
    
    #[derive(Default)]
    pub(super) struct ResourceLifetime {
        pub first_use: CombinedBatchIndex,
        pub last_use: CombinedBatchIndex
    }
    
    pub(super) struct PerPermTaskBuffer {
        /// Every permutation always has all buffers but they are not necessarily valid in that permutation.
        /// This boolean is used to check this.
        pub valid: bool,
        pub latest_access: Access,
        pub latest_access_batch_index: usize,
        pub latest_access_submit_scope_index: usize,
        pub first_access: Access,
        pub first_access_batch_index: usize,
        pub first_access_submit_scope_index: usize,
        // When the last index was a read and an additional read is followed after,
        // we will combine all barriers into one, which is the first barrier that the first read generates.
        pub latest_access_read_barrier_index: LastReadIndex,
        pub actual_buffer: BufferId,
        pub lifetime: ResourceLifetime,
        pub allocation_offset: usize
    }
    
    #[derive(Clone, Copy)]
    pub(super) struct ExtendedImageRangeState {
        pub state: ImageRangeState,
        pub latest_access_batch_index: usize,
        pub latest_access_submit_scope_index: usize,
        // When the last index was a read and an additional read is followed after,
        // we will combine all barriers into one, which is the first barrier that the first read generates.
        pub latest_access_read_barrier_index: LastReadIndex
    }
    
    #[derive(Default)]
    pub(super) struct PerPermTaskImage {
        /// Every permutation always has all buffers but they are not necessarily valid in that permutation.
        /// This boolean is used to check this.
        pub valid: bool,
        pub swapchain_semaphore_waited_upon: bool,
        pub last_range_states: Vec<ExtendedImageRangeState>,
        pub first_range_states: Vec<ExtendedImageRangeState>,
        pub lifetime: ResourceLifetime,
        pub usage: ImageUsageFlags,
        pub actual_image: ImageId,
        pub allocation_offset: usize
    }
    
    #[derive(Default)]
    pub(super) struct TaskBarrier {
        // When this ID is None, this is a memory barrier. When this ID is Some, this is an image memory barrier.
        pub image_id: Option<TaskImageHandle>,
        pub range: ImageSubresourceRange,
        pub src_layout: ImageLayout,
        pub dst_layout: ImageLayout,
        pub src_access: Access,
        pub dst_access: Access,
    }
    impl From<&TaskSplitBarrier> for TaskBarrier {
        fn from(value: &TaskSplitBarrier) -> Self {
            TaskBarrier { 
                image_id: value.image_id, 
                range: value.range, 
                src_layout: value.src_layout, 
                dst_layout: value.dst_layout, 
                src_access: value.src_access, 
                dst_access: value.dst_access 
            }
        }
    }
    
    pub(super) struct TaskSplitBarrier {
        pub split_barrier_state: SplitBarrierState,
        // When this ID is None, this is a memory barrier. When this ID is Some, this is an image memory barrier.
        pub image_id: Option<TaskImageHandle>,
        pub range: ImageSubresourceRange,
        pub src_layout: ImageLayout,
        pub dst_layout: ImageLayout,
        pub src_access: Access,
        pub dst_access: Access,
    }
    

    pub(super) struct Task {
        pub task: Box<dyn BaseTask>,
        // pub constant_buffer_size: u32,
        // pub use_offsets: (Vec<u32>, Vec<u32>), // (BufferOffsets, ImageOffsets)
        pub image_view_cache: Vec<Vec<ImageViewId>>
    }
    
    pub(super) struct PresentInfoInternal {
        pub binary_semaphores: Vec<BinarySemaphore>,
        pub additional_binary_semaphores: Vec<BinarySemaphore>
    }
    
    #[derive(Default)]
    pub(super) struct TaskBatch {
        pub pipeline_barrier_indices: Vec<usize>,
        pub wait_split_barrier_indices: Vec<usize>,
        pub tasks: Vec<TaskId>,
        pub signal_split_barrier_indices: Vec<usize>
    }
    
    #[derive(Default)]
    pub(super) struct TaskBatchSubmitScope {
        pub submit_info: CommandSubmitInfo,
        pub user_submit_info: TaskSubmitInfo,
        // These barriers are inserted after all batches and their sync
        pub last_minute_barrier_indices: Vec<usize>,
        pub task_batches: Vec<TaskBatch>,
        pub used_swapchain_task_images: Vec<u64>,
        pub present_semaphores: Option<(Vec<BinarySemaphore>, Vec<BinarySemaphore>)>
    }
    
    #[derive(Default)]
    pub(super) struct TaskGraphPermutation {
        // record time information
        pub active: bool,
        // persistent information
        pub swapchain_image: Option<TaskImageHandle>,
        pub buffer_infos: Vec<PerPermTaskBuffer>,
        pub image_infos: Vec<PerPermTaskImage>,
        pub split_barriers: Vec<TaskSplitBarrier>,
        pub barriers: Vec<TaskBarrier>,
        pub initial_barriers: Vec<usize>,
        // TODO(msakmary, pahrens) - Instead of storing batch submit scopes which contain batches
        // we should make a vector of batches and a second vector of submit scopes which are
        // just offsets into the batches vector
        pub batch_submit_scopes: Vec<TaskBatchSubmitScope>,
        pub swapchain_image_first_use_submit_scope_index: usize,
        pub swapchain_image_last_use_submit_scope_index: usize
    }

    fn update_buffer_first_access(buffer: &mut PerPermTaskBuffer, new_access_batch: usize, new_access_submit_scope: usize, new_access: Access) {
        use ash::vk::AccessFlags2;
        if buffer.first_access.1 == AccessFlags2::NONE {
            buffer.first_access = new_access;
            buffer.first_access_batch_index = new_access_batch;
            buffer.first_access_submit_scope_index = new_access_submit_scope;
        } else if buffer.first_access.1 == AccessFlags2::MEMORY_READ && new_access.1 == AccessFlags2::MEMORY_READ {
            buffer.first_access = buffer.first_access | new_access;
            let new_is_earlier = new_access_submit_scope < buffer.first_access_submit_scope_index
                                    || (new_access_submit_scope == buffer.first_access_submit_scope_index && new_access_batch < buffer.first_access_batch_index);
            if new_is_earlier {
                buffer.first_access_batch_index = new_access_batch;
                buffer.first_access_submit_scope_index = new_access_submit_scope;
            }
        }
    }
    
    impl TaskGraphPermutation {
        pub fn add_task(&mut self, task_id: TaskId, task_graph: &mut TaskGraph, task: &impl BaseTask) {
            // Set persistent task resources to be valid for the permutation.
            let resource_uses = task.get_resource_uses();
            for buffer_use in resource_uses.0 {
                if let TaskBufferHandle::Persistent { index } = buffer_use.handle {
                    self.buffer_infos[index as usize].valid = true;
                }
            }
            for image_use in resource_uses.1 {
                if let TaskImageHandle::Persistent { index, range } = image_use.handle {
                    self.image_infos[index as usize].valid = true;
                }
            }

            let current_submit_scope_index = self.batch_submit_scopes.len() - 1;

            // All tasks are reordered while recording.
            // Tasks are grouped into "task batches" which are just a group of tasks,
            // that can execute together while overlapping without synchronization between them.
            // Task batches are further grouped into submit scopes.
            // A submit scopes contains a group of batches between two submits.
            // At first, we find the batch we need to insert the new task into.
            // To optimize for optimal overlap and minimal pipeline barriers, we try to insert the task as early as possible.
            let batch_index = task_graph.schedule_task(
                self,
                current_submit_scope_index,
                task
            );
            // Add the task to the batch.
            self.batch_submit_scopes[current_submit_scope_index]
                .task_batches[batch_index]
                .tasks.push(task_id);

            // Now that we know what batch we need to insert the task into, we need to add synchronization between batches.
            // As stated earlier batches are groups of tasks which can execute together without sync between them.
            // To simplify and optimize the sync placement daxa only synchronizes between batches.
            // This effectively means that all the resource uses, and their memory and execution dependencies in a batch
            // are combined into a single unit which is synchronized against other batches.
            for buffer_use in resource_uses.0 {
                let handle_index = buffer_use.handle.index();
                let task_buffer = &mut self.buffer_infos[handle_index as usize];
                let current_buffer_access = task_buffer_access_to_access(&buffer_use.access());
                update_buffer_first_access(task_buffer, batch_index, current_submit_scope_index, current_buffer_access);
                // For transient buffers, we need to record first and last use so that we can later name their allocations.
                // TODO(msakmary, pahrens) We should think about how to combine this with update_buffer_first_access below since
                // they both overlap in what they are doing
                if let TaskBufferHandle::Transient { task_graph_index, index } = buffer_use.handle {
                    let buffer_first_use = &mut task_buffer.lifetime.first_use;
                    let buffer_last_use = &mut task_buffer.lifetime.last_use;
                    if current_submit_scope_index < buffer_first_use.submit_scope_index {
                        buffer_first_use.submit_scope_index = current_submit_scope_index;
                        buffer_last_use.submit_scope_index = batch_index;
                    } else if current_submit_scope_index == buffer_first_use.submit_scope_index {
                        buffer_first_use.task_batch_index;
                    }

                    if current_submit_scope_index > buffer_last_use.submit_scope_index || buffer_last_use.submit_scope_index == usize::MAX {
                        buffer_last_use.submit_scope_index = current_submit_scope_index;
                        buffer_last_use.task_batch_index = batch_index;
                    } else if current_submit_scope_index == buffer_last_use.submit_scope_index {
                        buffer_last_use.task_batch_index = usize::max(buffer_last_use.task_batch_index, batch_index);
                    }
                }
                // When the last use was a read AND the new use of the buffer is a read,
                // we need to add our stage flags to the existing barrier of the last use.
                use ash::vk::AccessFlags2;
                let is_last_access_read = task_buffer.latest_access.1 == AccessFlags2::MEMORY_READ;
                let is_current_access_read = current_buffer_access.1 == AccessFlags2::MEMORY_READ;
                // We only need barriers between two accesses.
                // If the previous access is none, the current access is the first access.
                // Therefore we do not need to insert any synchronization if the previous access is none.
                // This is buffer specific. Images have a layout that needs to be set from undefined to the current accesses layout.
                // When the latest access  is a read that did not require a barrier before we also do not need a barrier now.
                // So skip, if the latest access is read and there is no latest_access_read_barrier_index present.
                let is_last_access_none = task_buffer.latest_access.1 == AccessFlags2::NONE;
                if !is_last_access_none && !(task_buffer.latest_access_read_barrier_index == LastReadIndex::None && is_last_access_read) {
                    if is_last_access_read && is_current_access_read {
                        match task_buffer.latest_access_read_barrier_index {
                            LastReadIndex::Barrier(index) => {
                                let last_read_split_barrier = &mut self.split_barriers[index];
                                last_read_split_barrier.dst_access = last_read_split_barrier.dst_access | current_buffer_access;
                            },
                            LastReadIndex::SplitBarrier(index) => {
                                let last_read_barrier = &mut self.barriers[index];
                                last_read_barrier.dst_access = last_read_barrier.dst_access | current_buffer_access
                            },
                            _ => ()
                        }
                    } else {
                        // When the uses are incompatible (no read on read) we need to insert a new barrier.
                        // Host access needs to be handled in a specialized way.
                        use ash::vk::PipelineStageFlags2;
                        let src_host_only_access = task_buffer.latest_access.0 == PipelineStageFlags2::HOST;
                        let dst_host_only_access = current_buffer_access.0 == PipelineStageFlags2::HOST;
                        debug_assert!(!(src_host_only_access && dst_host_only_access), "Direct sync between two host accesses on GPU are not allowed!");
                        let is_host_barrier = src_host_only_access || dst_host_only_access;
                        // When the distance between src and dst batch is one, we can replace the split barrier with a normal barrier.
                        // We also need to make sure we do not use split barriers when the src or dst stage exclusively uses the host stage.
                        // This is because the host stage does not declare an execution dependency on the cpu but only a memory dependency.
                        let use_pipeline_barrier = (task_buffer.latest_access_batch_index + 1 == batch_index && current_submit_scope_index == task_buffer.latest_access_submit_scope_index) || is_host_barrier;
                        if use_pipeline_barrier {
                            let barrier_index = self.barriers.len();
                            self.barriers.push(TaskBarrier {
                                image_id: None, // This is not an image barrier.
                                src_access: task_buffer.latest_access,
                                dst_access: current_buffer_access,
                                ..Default::default()
                            });
                            // And we insert the barrier index into the list of pipeline barriers of the current tasks batch.
                            let batch = &mut self.batch_submit_scopes[current_submit_scope_index].task_batches[batch_index];
                            batch.pipeline_barrier_indices.push(barrier_index);
                            if current_buffer_access.1 == AccessFlags2::MEMORY_READ {
                                // As the new access is a read we remember our barrier index,
                                // So that potential future reads after this can reuse this barrier.
                                task_buffer.latest_access_read_barrier_index = LastReadIndex::Barrier(barrier_index);
                            }
                        } else {
                            let split_barrier_index = self.split_barriers.len();
                            self.split_barriers.push(TaskSplitBarrier {
                                split_barrier_state: task_graph.device.create_split_barrier(SplitBarrierInfo {
                                    debug_name: format!("TaskGraph \"{}\" SplitBarrier {}", task_graph.info.debug_name, split_barrier_index).into()
                                }).unwrap(),
                                image_id: None, // This is not an image barrier.
                                range: Default::default(),
                                src_layout: Default::default(),
                                dst_layout: Default::default(),
                                src_access: task_buffer.latest_access,
                                dst_access: current_buffer_access,
                            });
                            // Now we give the src batch the index of this barrier to signal.
                            let src_scope = &mut self.batch_submit_scopes[task_buffer.latest_access_submit_scope_index];
                            let src_batch = &mut src_scope.task_batches[task_buffer.latest_access_batch_index];
                            src_batch.signal_split_barrier_indices.push(split_barrier_index);
                            // And we also insert the split barrier index into the waits of the current task's batch.
                            let batch = &mut self.batch_submit_scopes[current_submit_scope_index].task_batches[batch_index];
                            batch.wait_split_barrier_indices.push(split_barrier_index);
                            if current_buffer_access.1 == AccessFlags2::MEMORY_READ {
                                // As the new access is a read, we remember our barrier index,
                                // so that potential future reads after this can reuse this barrier.
                                task_buffer.latest_access_read_barrier_index = LastReadIndex::SplitBarrier(split_barrier_index);
                            } else {
                                task_buffer.latest_access_read_barrier_index = LastReadIndex::None
                            }
                        }
                    }
                }
                // Now that we inserted/updated the synchronization, we update the latest access.
                task_buffer.latest_access = current_buffer_access;
                task_buffer.latest_access_batch_index = batch_index;
                task_buffer.latest_access_submit_scope_index = current_submit_scope_index;
            }
            let mut tracked_range_rests: Vec<ExtendedImageRangeState> = vec![];
            let mut new_use_ranges: Vec<ImageSubresourceRange> = vec![];
            for image_use in resource_uses.1 {
                let used_image_t_id = &image_use.handle;
                let used_image_t_access = &image_use.access;
                let initial_used_image_range = used_image_t_id.range();
                let used_image_index = used_image_t_id.index();
                let task_image = &mut self.image_infos[used_image_index as usize];
                // For transient images we need to record first and last use so that we can later name their allocations
                // TODO(msakmary, pahrens) We should think about how to combine this with update_image_inital_slices below since
                // they both overlap in what they are doing
                if let PermIndepTaskImageInfo::Transient { info, memory_requirements } = &task_graph.global_image_infos[used_image_index as usize] {
                    let image_first_use = &mut task_image.lifetime.first_use;
                    let image_last_use = &mut task_image.lifetime.last_use;
                    if current_submit_scope_index < image_first_use.submit_scope_index {
                        image_first_use.submit_scope_index = current_submit_scope_index;
                        image_first_use.task_batch_index = batch_index;
                    } else if current_submit_scope_index == image_first_use.submit_scope_index {
                        image_first_use.task_batch_index = usize::min(image_first_use.task_batch_index, batch_index);
                    }

                    if current_submit_scope_index > image_last_use.submit_scope_index || image_last_use.submit_scope_index == usize::MAX {
                        image_last_use.submit_scope_index = current_submit_scope_index;
                        image_last_use.task_batch_index = batch_index;
                    } else if current_submit_scope_index == image_last_use.submit_scope_index {
                        image_last_use.task_batch_index = usize::max(image_last_use.task_batch_index, batch_index);
                    }
                }
                task_image.usage |= task_image_access_to_usage(used_image_t_access);
                let (current_image_layout, current_image_access) = task_image_access_to_layout_access(used_image_t_access);
                // Now this seems strange, why would be need multiple current use slices, as we only have one here.
                // This is because when we intersect this slice with the tracked slices, we get an intersection and a rest.
                // We need to then test the rest against all the remaining tracked uses,
                // as the intersected part is already beeing handled in the following code.
                new_use_ranges.push(initial_used_image_range.clone());
                // This is the tracked slice we will insert after we finished analyzing the current used image.
                let mut ret_new_use_tracked_range = ExtendedImageRangeState {
                    state: ImageRangeState {
                        latest_access: current_image_access,
                        latest_layout: current_image_layout,
                        range: initial_used_image_range.clone()
                    },
                    latest_access_batch_index: batch_index,
                    latest_access_submit_scope_index: current_submit_scope_index,
                    latest_access_read_barrier_index: LastReadIndex::None // This is a dummy value (either set later or ignored entirely).
                };
                // We update the initial access ranges.
                update_image_initial_access_ranges(task_image, &mut ret_new_use_tracked_range);
                // As image subresources can be in different layouts and also different synchronization scopes,
                // we need to track these image ranges individually.
                let mut tracked_range_index = 0;
                while tracked_range_index != task_image.last_range_states.len() {
                    let mut advanced_tracked_range_index = false;
                    let mut used_image_range_index = 0;
                    while used_image_range_index != new_use_ranges.len() {
                        // We make a local copy of both ranges here.
                        // We cannot rely on indexing the vectors, as we modify them in this function.
                        // For this inner loop we want to remember the information about these ranges,
                        // even after they are removed from their respective vector.
                        let used_image_range = new_use_ranges[used_image_range_index];
                        let tracked_range = task_image.last_range_states[tracked_range_index];
                        // We are only interested in intersecting ranges, as use of non intersecting ranges does not need synchronization.
                        if !used_image_range.intersects(tracked_range.state.range) {
                            // We only need to advance the iterator manually here.
                            // After this if statement there is an unconditional erase that advances the iterator if this is not hit.
                            used_image_range_index += 1;
                            continue;
                        }
                        // As we found an intersection, part of the old tracked range must be altered.
                        // Instead of altering it, we remove it and add the rest ranges back in later.
                        new_use_ranges.remove(used_image_range_index);
                        // Now we know that the new use intersects ranges with a previous use.
                        // This means that we need to find the intersecting part,
                        // sync the intersecting part from the previous use to the current use
                        // and remove the overlapping part from the tracked range.
                        // Now we need to split the uses into three groups:
                        // * the range that is the intersection of the tracked image range and the current new use (intersection)
                        // * the part of the tracked image range that does not intersect with the current new use (tracked_range_rest)
                        // * the part of the current new use range that does not intersect with the tracked image (new_use_range_rest)
                        let intersection = tracked_range.state.range.intersect(used_image_range);
                        let (tracked_range_rest, tracked_range_rest_count) = tracked_range.state.range.subtract(intersection);
                        let (new_use_range_rest, new_use_range_rest_count) = used_image_range.subtract(intersection);
                        // We now remove the old tracked range from the list of tracked ranges, as we just split it.
                        // We need to know if the iterator was advanced. This erase advances the iterator.
                        // If the iterator was not advanced by this we need to advance it ourself manually later.
                        advanced_tracked_range_index = true;
                        task_image.last_range_states.remove(tracked_range_index);
                        // Now we remember the left over range from the original tracked range.
                        for rest_index in 0..tracked_range_rest_count {
                            // The rest tracked ranges are the same as the original tracked range,
                            // except for the range intself, which is the remainder of the subtraction of the intersection.
                            let mut current_rest_tracked_range = tracked_range;
                            current_rest_tracked_range.state.range = tracked_range_rest[rest_index];
                            tracked_range_rests.push(current_rest_tracked_range);
                        }
                        // Now we remember the left over range from our current used range.
                        for rest_index in 0..new_use_range_rest_count {
                            // We reassign the index here as it is getting invalidated by the insert.
                            // The new index points to the newly inserted element.
                            // When the next iteration of the outer for loop starts, all these elements get skipped.
                            // This is good as these elements do NOT intersect with the currently inspected tracked range.
                            used_image_range_index = new_use_ranges.len();
                            new_use_ranges.insert(new_use_ranges.len(), new_use_range_rest[rest_index]);
                        }
                        // Every other access (NONE, READ_WRITE, WRITE) are interpreted as writes in this context.
                        // When the last use was a read AND the new use of the image is a read,
                        // we need to add our stageflags to the existing barrier of the last use.
                        // To be able to do this the layout of the image range must also match.
                        // If they differ, we needto insert an execution barrier with a layout transition.
                        use ash::vk::PipelineStageFlags2;
                        use ash::vk::AccessFlags2;
                        let is_last_access_read = tracked_range.state.latest_access.1 == AccessFlags2::MEMORY_READ;
                        let is_current_access_read = current_image_access.1 == AccessFlags2::MEMORY_READ;
                        let are_layouts_identical = tracked_range.state.latest_layout == current_image_layout;
                        if is_last_access_read && is_current_access_read && are_layouts_identical {
                            match tracked_range.latest_access_read_barrier_index {
                                LastReadIndex::SplitBarrier(index) => {
                                    let last_read_split_barrier = &mut self.split_barriers[index];
                                    last_read_split_barrier.dst_access = last_read_split_barrier.dst_access | tracked_range.state.latest_access;
                                },
                                LastReadIndex::Barrier(index) => {
                                    let last_read_barrier = &mut self.barriers[index];
                                    last_read_barrier.dst_access = last_read_barrier.dst_access | tracked_range.state.latest_access;
                                },
                                _ => ()
                            }
                        } else {
                            // When the uses are incompatible (no read on read, or no identical layout) we need to insert a new barrier.
                            // Host access needs to be handled in a specialized way.
                            let src_host_only_access = tracked_range.state.latest_access.0 == PipelineStageFlags2::HOST;
                            let dst_host_only_access = tracked_range.state.latest_access.0 == PipelineStageFlags2::HOST;
                            debug_assert!(!(src_host_only_access && dst_host_only_access), "Direct sync between two host accesses on GPU is not allowed!");
                            let is_host_barrier = src_host_only_access || dst_host_only_access;
                            // When the distance between src and dst batch is one, we can replace the split barrier with a normal barrier.
                            // We also need to make sure we do not use split barriers when the src or dst stage exclusively uses the host stage.
                            // This is because the host stage does not declare an execution dependency on the cpu but only a memory dependency.
                            let use_pipeline_barrier = (tracked_range.latest_access_batch_index + 1 == batch_index
                                                            && current_submit_scope_index == tracked_range.latest_access_submit_scope_index)
                                                            || is_host_barrier;
                            if use_pipeline_barrier {
                                let barrier_index = self.barriers.len();
                                self.barriers.push(TaskBarrier {
                                    image_id: Some(*used_image_t_id),
                                    range: intersection,
                                    src_layout: tracked_range.state.latest_layout,
                                    dst_layout: current_image_layout,
                                    src_access: tracked_range.state.latest_access,
                                    dst_access: current_image_access,
                                });
                                // And we insert the barrier index into the list of pipeline barriers of the current task's batch.
                                let batch = &mut self.batch_submit_scopes[current_submit_scope_index].task_batches[batch_index];
                                batch.pipeline_barrier_indices.push(barrier_index);
                                if current_image_access.1 == AccessFlags2::MEMORY_READ {
                                    // As the new access is a read, we remember out barrier index
                                    // so that potential future reads after this can reuse this barrier.
                                    ret_new_use_tracked_range.latest_access_read_barrier_index = LastReadIndex::Barrier(barrier_index);
                                }
                            } else {
                                let split_barrier_index = self.split_barriers.len();
                                self.split_barriers.push(TaskSplitBarrier {
                                    split_barrier_state: task_graph.device.create_split_barrier(SplitBarrierInfo {
                                        debug_name: format!("TaskGraph \"{}\" SplitBarrier (Image) {}", task_graph.info.debug_name, split_barrier_index).into(),
                                    }).unwrap(),
                                    image_id: Some(*used_image_t_id),
                                    range: intersection,
                                    src_layout: tracked_range.state.latest_layout,
                                    dst_layout: current_image_layout,
                                    src_access: tracked_range.state.latest_access,
                                    dst_access: current_image_access,
                                });
                                // Now we give the src batch the index of this barrier to signal.
                                let src_scope = &mut self.batch_submit_scopes[tracked_range.latest_access_batch_index];
                                let src_batch = &mut src_scope.task_batches[tracked_range.latest_access_batch_index];
                                src_batch.signal_split_barrier_indices.push(split_barrier_index);
                                // And we also insert the split barrier index into the waits of the current task's batch.
                                let batch = &mut self.batch_submit_scopes[current_submit_scope_index].task_batches[batch_index];
                                batch.wait_split_barrier_indices.push(split_barrier_index);
                                if current_image_access.1 == AccessFlags2::MEMORY_READ {
                                    // As the new access is a read, we remember our barrier index
                                    // so that potential future reads after this can reuse this barrier.
                                    ret_new_use_tracked_range.latest_access_read_barrier_index = LastReadIndex::SplitBarrier(split_barrier_index);
                                } else {
                                    ret_new_use_tracked_range.latest_access_read_barrier_index = LastReadIndex::None;
                                }
                            }
                        }
                        // Make sure we have any tracked slices to intersect with left.
                        if tracked_range_index == task_image.last_range_states.len() {
                            break;
                        }
                    }
                    if !advanced_tracked_range_index {
                        // If we didn't find any intersections, we don't remove the tracked slice.
                        // Removing a tracked slice "advances" the index. As we did not remove,
                        // we need to advance it manually.
                        tracked_range_index += 1;
                    }
                }
                // Now we need to add the latest use and tracked range of our current access.
                task_image.last_range_states.push(ret_new_use_tracked_range);
                // The remainder tracked ranges we remembered from earlier are now inserted back in to the list of tracked ranges.
                // We deferred this step as we don't want to check these in the loop above, as we found them to not intersect with the new use.
                task_image.last_range_states.append(&mut tracked_range_rests);
            }
        }
    
        pub fn submit(info: &TaskSubmitInfo) {
            todo!()
        }
    
        pub fn present(info: &TaskPresentInfo) {
            todo!()
        }
    }
    
    pub(super) struct PersistentTaskBuffer {
        pub info: TaskBufferInfo,
        // One task buffer can back multiple buffers.
        pub actual_buffers: Vec<BufferId>,
        // We store runtime information about the previous executions final resource states.
        // This is important, as with conditional execution and temporal resources we need to store this infomation to form correct state transitions.
        pub latest_access: Access,
    
        pub unique_index: u32
    }
    
    // Used to allocate id - because all persistent resources have unique id we need a single point
    // from which they are generated
    static exec_unique_next_buffer_index: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    impl PersistentTaskBuffer {
        pub fn new(info: TaskBufferInfo) -> Self {
            todo!()
        }
    }

    impl Drop for PersistentTaskBuffer {
        fn drop(&mut self) {
            todo!()
        }
    }

    pub(super) struct PersistentTaskImage {
        pub info: TaskImageInfo,
        // One task image can back multiple images.
        pub actual_images: Vec<ImageId>,
        // We store runtime information about the previous executions final resource states.
        // This is important, as with conditional execution and temporal resources we need to store this infomation to form correct state transitions.
        pub latest_range_states: Vec<ImageRangeState>,

        pub unique_index: u32
    }

    static exec_unique_next_image_index: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    impl PersistentTaskImage {
        pub fn new(info: TaskImageInfo) -> Self {
            todo!()
        }
    }

    impl Drop for PersistentTaskImage {
        fn drop(&mut self) {
            todo!()
        }
    }

    pub(super) enum PermIndepTaskBufferInfo {
        Persistent {
            buffer: PersistentTaskBuffer
        },
        Transient {
            info: TaskTransientBufferInfo,
            memory_requirements: ash::vk::MemoryRequirements
        }
    }

    impl PermIndepTaskBufferInfo {
        pub fn get_name(&self) -> String {
            match self {
                PermIndepTaskBufferInfo::Persistent { buffer } => {
                    buffer.info.name.clone()
                },
                PermIndepTaskBufferInfo::Transient { info, memory_requirements } => {
                    info.name.clone()
                }
            }
        }
    }

    pub(super) enum PermIndepTaskImageInfo {
        Persistent {
            image: PersistentTaskImage
        },
        Transient {
            info: TaskTransientImageInfo,
            memory_requirements: ash::vk::MemoryRequirements
        }
    }

    impl PermIndepTaskImageInfo {
        pub fn get_name(&self) -> String {
            match self {
                PermIndepTaskImageInfo::Persistent { image } => {
                    image.info.name.clone()
                },
                PermIndepTaskImageInfo::Transient { info, memory_requirements } => {
                    info.name.clone()
                }
            }
        }
    }
    
    // TaskGraph internal methods
    impl super::TaskGraph {
        fn translate_persistent_ids(&mut self, task: &mut impl BaseTask) {
            let (buffer_uses, image_uses) = task.get_resource_uses_mut();

            for (i, buffer_use) in buffer_uses.iter_mut().enumerate() {
                // match buffer_use.handle {
                //     TaskBufferHandle::Empty => {
                //         #[cfg(debug_assertions)]
                //         panic!(
                //             "Detected empty task buffer handle in use (index: {}, access: {:?}) in task \"{}\"",
                //             i, buffer_use.access(), task.get_name()
                //         );
                //         #[cfg(not(debug_assertions))]
                //         unreachable!();
                //     },
                //     _ => ()
                // }
                buffer_use.handle = self.buffer_id_to_local_id(buffer_use.handle);
            }

            for (i, image_use) in image_uses.iter_mut().enumerate() {
                // match image_use.handle {
                //     TaskImageHandle::Empty => {
                //         #[cfg(debug_assertions)]
                //         panic!(
                //             "Detected empty task image handle in use (index: {}, access: {:?}) in task \"{}\"",
                //             i, image_use.access(), task.get_name()
                //         );
                //         #[cfg(not(debug_assertions))]
                //         unreachable!();
                //     },
                //     _ => ()
                // }
                image_use.handle = self.image_id_to_local_id(image_use.handle);
            }
        }

        fn get_task_arg_shader_offsets_size(&self, args: (&[TaskBufferUse], &[TaskImageUse])) -> ((Vec<u32>, Vec<u32>), u32) {
            let mut ret = (vec![], vec![]);
            ret.0.reserve(args.0.len());
            ret.1.reserve(args.1.len());
            let mut offset = 0;
            // Buffer uses
            let align = 8;
            for _ in args.0 { 
                offset = (offset + align - 1) / align * align;
                ret.0.push(offset);
                offset += align;
            }
            // Image uses
            let align = 4;
            for _ in args.1 { 
                offset = (offset + align - 1) / align * align;
                ret.1.push(offset);
                offset += align;
            }

            (ret, offset)
        }

        pub(super) fn add_task_internal(&mut self, mut task: impl BaseTask + 'static) {
            self.translate_persistent_ids(&mut task);
            // Overlapping resource uses can be valid in the case of reads in the same layout for example.
            // But in order to make the task list implementation simpler,
            // daxa does not allow for overlapping use of a resource within a task, 
            // even when it is a read in the same layout.
            #[cfg(debug_assertions)]
            self.check_for_overlapping_use(&task);

            let task_id = self.tasks.len();

            let mut permutations = std::mem::take(&mut self.permutations);
            for permutation in permutations.iter_mut()  {
                permutation.add_task(task_id, self, &task);
            }
            self.permutations = permutations;

            let resource_uses = task.get_resource_uses();
            // let mut constant_buffer_use_offsets: (Vec<u32>, Vec<u32>) = Default::default();
            // let mut constant_buffer_size: u32 = 0;
            // if task.get_uses_constant_buffer_slot() != -1 {
            //     (constant_buffer_use_offsets, constant_buffer_size) = self.get_task_arg_shader_offsets_size(resource_uses);
            // }

            let mut view_cache: Vec<Vec<ImageViewId>> = Default::default();
            view_cache.resize(resource_uses.1.len(), Default::default());
            self.tasks.push(Task {
                task: Box::new(task),
                // constant_buffer_size,
                // use_offsets: constant_buffer_use_offsets,
                image_view_cache: view_cache
            });
        }

        fn schedule_task(
            &mut self,
            permutation: &mut TaskGraphPermutation,
            current_submit_scope_index: usize,
            task: & impl BaseTask
        ) -> usize {
            let current_submit_scope = &mut permutation.batch_submit_scopes[current_submit_scope_index];

            let mut first_possible_batch_index = 0;
            if !self.info.reorder_tasks {
                first_possible_batch_index = usize::max(current_submit_scope.task_batches.len(), 1) - 1;
            }
    
            let resource_uses = task.get_resource_uses();
            for buffer_use in resource_uses.0 {
                let handle_index = buffer_use.handle.index();
                let task_buffer = &permutation.buffer_infos[handle_index as usize];
                // If the latest access is in a previous submit scope, the earliest batch we can insert into is
                // the current scopes first batch.
                if task_buffer.latest_access_submit_scope_index < current_submit_scope_index { continue };
    
                let current_buffer_access = task_buffer_access_to_access(&buffer_use.access());
                // Every other access (NONE, READ_WRITE, WRITE) are interpreted as writes in this context.
                use ash::vk::AccessFlags2;
                let is_last_access_read = task_buffer.latest_access.1 == AccessFlags2::MEMORY_READ;
                let is_last_access_none = task_buffer.latest_access.1 == AccessFlags2::NONE;
                let is_current_access_read = current_buffer_access.1 == AccessFlags2::MEMORY_READ;
    
                // TODO(msakmarry): improve sheduling here to reorder reads in front of each other, respecting the last to read barrier if present!
                // When a buffer has been read in a previous use AND the current task also reads the buffer,
                // we must insert the task at or after the last use batch.
                let mut current_buffer_first_possible_batch_index = task_buffer.latest_access_batch_index;
                // So when not both the last access and the current access are reads, we need to insert AFTER the latest access.
                if !(is_last_access_read && is_current_access_read) && !is_last_access_none {
                    current_buffer_first_possible_batch_index += 1;
                }
                first_possible_batch_index = usize::max(first_possible_batch_index, current_buffer_first_possible_batch_index);
            }
            for image_use in resource_uses.1 {
                let handle_index = image_use.handle.index();
                let task_image = &permutation.image_infos[handle_index as usize];
                let global_task_image = &self.global_image_infos[handle_index as usize];
    
                debug_assert!(!task_image.swapchain_semaphore_waited_upon, "Swapchain image is already presented!");
    
                if let PermIndepTaskImageInfo::Persistent { image } = global_task_image {
                    if image.info.swapchain_image {
                        if permutation.swapchain_image_first_use_submit_scope_index == usize::MAX {
                            permutation.swapchain_image_first_use_submit_scope_index = current_submit_scope_index;
                            permutation.swapchain_image_last_use_submit_scope_index = current_submit_scope_index;
                        } else {
                            permutation.swapchain_image_first_use_submit_scope_index = usize::min(current_submit_scope_index, permutation.swapchain_image_first_use_submit_scope_index);
                            permutation.swapchain_image_last_use_submit_scope_index = usize::max(current_submit_scope_index, permutation.swapchain_image_last_use_submit_scope_index);
                        }
                    }
                }

                let (this_task_image_layout, this_task_image_access) = task_image_access_to_layout_access(&image_use.access);
                // As image subresources can be in different layouts and also different synchronization scopes,
                // we need to track these image ranges individually.
                for tracked_range in &task_image.last_range_states {
                    // If the latest access is in a previous submit scope, the earliest batch we can insert into is
                    // the current scopes first batch.
                    // When the slices dont intersect, we dont need to do any sync or execution ordering between them.
                    let use_range = image_use.handle.range();
                    if tracked_range.latest_access_submit_scope_index < current_submit_scope_index || !tracked_range.state.range.intersects(use_range) {
                        continue;
                    }
                    // Now that we found out that the new use and an old use intersect,
                    // we need to insert the task in the same or a later batch.
                    use ash::vk::AccessFlags2;
                    let is_last_access_read = tracked_range.state.latest_access.1 == AccessFlags2::MEMORY_READ;
                    let is_current_access_read = this_task_image_access.1 == AccessFlags2::MEMORY_READ;
                    // When the image layouts differ, we must do a layout transition between reads.
                    // This forces us to place the task into a batch AFTER the tracked uses last batch.
                    let is_layout_identical = this_task_image_layout == tracked_range.state.latest_layout;
                    let mut current_image_first_possible_batch_index = tracked_range.latest_access_batch_index;
                    // If either the image layouts differ, or not both accesses are reads, we must place the task in a later batch.
                    if !(is_last_access_read && is_current_access_read && is_layout_identical) {
                        current_image_first_possible_batch_index += 1;
                    }
                    first_possible_batch_index = usize::max(first_possible_batch_index, current_image_first_possible_batch_index);
                }
            }
    
            // Make sure we have enough batches.
            if first_possible_batch_index >= current_submit_scope.task_batches.len() {
                current_submit_scope.task_batches.resize_with(first_possible_batch_index + 1, || TaskBatch::default());
            }
            
            first_possible_batch_index
        }

        pub(super) fn get_actual_buffers(&self, id: TaskBufferHandle, permutation: &TaskGraphPermutation) -> Box<[BufferId]> {
            match id {
                TaskBufferHandle::Persistent { index } => {
                    let global_buffer = &self.global_buffer_infos[index as usize];
                    match global_buffer {
                        PermIndepTaskBufferInfo::Persistent { buffer } => {
                            buffer.actual_buffers.as_slice().into()
                        },
                        _ => unreachable!()
                    }
                },
                TaskBufferHandle::Transient { task_graph_index, index } => {
                    let permutation_buffer = &permutation.buffer_infos[index as usize];
                    debug_assert!(permutation_buffer.valid, "Cannot get actual buffer, as this buffer is not valid in this permutation.");
                    [permutation_buffer.actual_buffer].into()
                }
            }
        }

        pub(super) fn get_actual_images(&self, id: TaskImageHandle, permutation: &TaskGraphPermutation) -> Box<[ImageId]> {
            match id {
                TaskImageHandle::Persistent { index, range } => {
                    let global_image = &self.global_image_infos[index as usize];
                    match global_image {
                        PermIndepTaskImageInfo::Persistent { image } => {
                            image.actual_images.as_slice().into()
                        },
                        _ => unreachable!()
                    }
                },
                TaskImageHandle::Transient { task_graph_index, index, range } => {
                    let permutation_image = &permutation.image_infos[index as usize];
                    debug_assert!(permutation_image.valid, "Cannot get actual image, as this image is not valid in this permutation.");
                    [permutation_image.actual_image].into()
                }
            }
        }

        pub(super) fn buffer_id_to_local_id(&self, id: TaskBufferHandle) -> TaskBufferHandle {
            match id {
                TaskBufferHandle::Persistent { index } => {
                    debug_assert!(
                        self.persistent_buffer_index_to_local_index.contains_key(&index),
                        "Detected invalid access of persistent task buffer id ({}) in task graph \"{}\".\nPlease make sure to declare persistent resource use to each task graph that uses this buffer with the function use_persistent_buffer!",
                        index, self.info.debug_name
                    );
                    TaskBufferHandle::Transient { task_graph_index: self.unique_index, index: self.persistent_buffer_index_to_local_index[&index] }
                },
                TaskBufferHandle::Transient { task_graph_index, index } => {
                    debug_assert!(
                        task_graph_index == self.unique_index,
                        "Detected invalid access of transient task buffer id ({}) in task graph\"{}\".\nPlease make sure that you only use transient buffers within the task graph they are created in!",
                        index, self.info.debug_name
                    );
                    TaskBufferHandle::Transient { task_graph_index: self.unique_index, index }
                }
            }
        }

        pub(super) fn image_id_to_local_id(&self, id: TaskImageHandle) -> TaskImageHandle {
            match id {
                TaskImageHandle::Persistent { index, range } => {
                    debug_assert!(
                        self.persistent_image_index_to_local_index.contains_key(&index),
                        "Detected invalid access of persistent task image id ({}) in task graph \"{}\".\nPlease make sure to declare persistent resource use to each task graph that uses this image with the function use_persistent_image!",
                        index, self.info.debug_name
                    );
                    TaskImageHandle::Transient { task_graph_index: self.unique_index, index, range }
                },
                TaskImageHandle::Transient { task_graph_index, index, range } => {
                    debug_assert!(
                        task_graph_index == self.unique_index,
                        "Detected invalid access of transient task image id ({}) in task graph \"{}\".\nPlease make sure that you only use transient images within the task graph they are created in!",
                        index, self.info.debug_name
                    );
                    TaskImageHandle::Transient { task_graph_index: self.unique_index, index, range }
                }
            }
        }

        pub(super) fn update_active_permutations(&mut self) {
            self.record_active_permutations.clear();

            for permutation_index in 0..self.permutations.len() {
                let active = (self.record_active_conditional_scopes & permutation_index as u32) == (self.record_active_conditional_scopes & self.record_conditional_states);
                self.permutations[permutation_index].active = active;
                if active {
                    self.record_active_permutations.push(permutation_index as u32);
                }
            }
        }

        pub(super) fn update_image_view_cache(&self, task: &mut Task, permutation: &TaskGraphPermutation) {
            #[cfg(debug_assertions)]
            let task_name = task.task.get_name();

            let uses = task.task.get_resource_uses_mut();
            // Image uses
            for (image_use_index, image_use) in uses.1.iter_mut().enumerate() {
                let range = match image_use.handle {
                    TaskImageHandle::Persistent { index, range } => range,
                    TaskImageHandle::Transient { task_graph_index, index, range } => range
                };
                // The image id here is already the task graph local id.
                // The persistent ids are converted to local ids in the add_task function.
                let task_id = image_use.handle;

                let actual_images = self.get_actual_images(task_id, permutation);
                let view_cache = &mut task.image_view_cache[image_use_index];

                let mut cache_valid = actual_images.len() == view_cache.len();
                if cache_valid {
                    for index in 0..actual_images.len() {
                        cache_valid = cache_valid && self.device.info_image_view(view_cache[index]).image == actual_images[index];
                    }
                }
                if !cache_valid {
                    #[cfg(debug_assertions)]
                    {
                        let (index, range) = match task_id {
                            TaskImageHandle::Persistent { index, range } => (index, range),
                            TaskImageHandle::Transient { task_graph_index, index, range } => (index, range)
                        };
                        self.validate_runtime_image_range(permutation, image_use_index, index as usize, &range);
                        self.validate_image_uses(permutation, image_use_index, index as usize, image_use.access, &task_name);
                    }

                    for &view in view_cache.iter() {
                        let parent_image_default_view = self.device.info_image_view(view).image.default_view();
                        // Cannot destroy the default view of an image!
                        if parent_image_default_view != view {
                            self.device.destroy_image_view(view);
                        }
                    }
                    view_cache.clear();
                    for index in 0..actual_images.len() {
                        let parent = actual_images[index];
                        let mut view_info = self.device.info_image_view(parent.default_view()).clone();
                        let use_view_type = image_use.view_type;

                        // When the use image view parameters match the default view,
                        // then use the default view id and avoid creating a new id here.
                        let is_use_default_range = view_info.subresource_range.equals(range);
                        let is_use_default_view_type = use_view_type == view_info.image_view_type;
                        if is_use_default_range && is_use_default_view_type {
                            view_cache.push(parent.default_view());
                        } else {
                            view_info.image_view_type = use_view_type;
                            view_info.subresource_range = range;
                            view_cache.push(self.device.create_image_view(view_info).unwrap());
                        }
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        fn validate_runtime_image_range(&self, permutation: &TaskGraphPermutation, use_index: usize, task_image_index: usize, access_range: &ImageSubresourceRange) {
            let actual_images = self.get_actual_images(TaskImageHandle::Transient { task_graph_index: self.unique_index, index: task_image_index as u32, range: Default::default() }, permutation);
            let task_name = self.global_image_infos[task_image_index].get_name();
            for index in 0..actual_images.len() {
                let full_range = self.device.info_image_view(actual_images[index].default_view()).subresource_range;
                let name = &self.device.info_image(actual_images[index]).debug_name;
                let use_within_runtime_image_counts = (access_range.base_mip_level + access_range.level_count <= full_range.base_mip_level + full_range.level_count)
                                                        && (access_range.base_array_layer + access_range.layer_count <= full_range.base_array_layer + full_range.layer_count);
                debug_assert!(
                    use_within_runtime_image_counts,
                    "Task image argument (arg index: {}, task image: \"{}\", range: {:?}) exceeds runtime image (index: {}, name: \"{}\" dimensions ({:?})!",
                    use_index, task_name, access_range, index, name, full_range
                )
            }
        }

        #[cfg(debug_assertions)]
        fn validate_image_uses(&self, permutation: &TaskGraphPermutation, use_index: usize, task_image_index: usize, task_access: TaskImageAccess, task_name: &String) {
            let use_flags = task_image_access_to_usage(&task_access);
            let actual_images = self.get_actual_images(TaskImageHandle::Transient { task_graph_index: self.unique_index, index: task_image_index as u32, range: Default::default() }, permutation);
            let task_image_name = self.global_image_infos[task_image_index].get_name();
            for index in 0..actual_images.len() {
                let image = actual_images[index];
                let access_valid = self.device.info_image(image).usage.contains(use_flags);
                debug_assert!(
                    access_valid,
                    "Detected invalid runtime image \"{}\" of task image \"{}\", in use {} of task \"{}\".\nThe given runtime image does NOT have the image use flag {:?} set, but the task use requires this use for all runtime images!",
                    self.device.info_image(image).debug_name, task_image_name, use_index, task_name, use_flags
                )
            }
        }

        pub(super) fn execute_task(&mut self, runtime: &mut TaskRuntimeInterface, permutation: &TaskGraphPermutation, in_batch_task_index: TaskBatchId, task_id: TaskId) {
            // We always allow reuse of the last command list ONCE within the task callback.
            // When the get_command_list function is called in a task, this is set to false.
            runtime.reuse_last_command_list = false;

            let mut tasks = std::mem::take(&mut self.tasks);
            let task = &mut tasks[task_id];
            self.update_image_view_cache(task, permutation);

            let uses = task.task.get_resource_uses_mut();
            // Buffer uses
            for buffer_use in uses.0.iter_mut() {
                buffer_use.buffers = self.get_actual_buffers(buffer_use.handle, permutation);
            }
            // Image uses
            for (index, image_use) in uses.1.iter_mut().enumerate() {
                image_use.images = self.get_actual_images(image_use.handle, permutation);
                image_use.views = task.image_view_cache[index].as_slice().into()
            }

            // TODO: Constant buffer support
            //let upload_args_to_constant_buffer = 

            runtime.current_task = Some(task_id);
            runtime.command_lists.last_mut().unwrap().begin_label(CommandLabelInfo {
                label_name: format!("Task {} \"{}\"", in_batch_task_index, task.task.get_name()).into(),
                label_color: self.info.task_label_color
            });
            task.task.callback(&TaskInterface {
                uses: TaskInterfaceUses { backend: runtime },
                backend: runtime
            });
            runtime.command_lists.last_mut().unwrap().end_label();

            // Return temporarily owned fields
            self.tasks = tasks;
        }

        pub(super) fn insert_pre_batch_barriers(&self, permutation: &TaskGraphPermutation) {
            todo!()
        }

        #[cfg(debug_assertions)]
        pub(super) fn check_for_overlapping_use(&self, task: &impl BaseTask) {
            let uses = task.get_resource_uses();
            // Buffer uses
            for (index_a, use_a) in uses.0.iter().enumerate() {
                for (index_b, use_b) in uses.0.iter().enumerate() {
                    if index_a == index_b { continue };

                    let overlapping = use_a.handle == use_b.handle;
                    let handle_index = match use_a.handle {
                        TaskBufferHandle::Persistent { index } => index,
                        TaskBufferHandle::Transient { task_graph_index, index } => index,
                        _ => unreachable!()
                    };
                    debug_assert!(
                        !overlapping, 
                        "Detected overlapping uses (input index {} and {}) of buffer \"{}\" in task \"{}\";\n All buffer uses must be disjoint in each task!",
                        index_a, index_b,
                        self.global_buffer_infos[handle_index as usize].get_name(),
                        task.get_name()
                    );
                }
            }
            // Image uses
            for (index_a, use_a) in uses.1.iter().enumerate() {
                for (index_b, use_b) in uses.1.iter().enumerate() {
                    if index_a == index_b { continue };

                    let range_a = use_a.handle.range();
                    let range_b = use_b.handle.range();

                    let intersect = use_a.handle == use_b.handle && range_a.intersects(range_b);
                    let intersection = range_a.intersect(range_b);
                    let handle_index = use_a.handle.index();
                    debug_assert!(
                        !intersect,
                        "Detected range overlap between task image uses\n(index: {}, slice: ({:?})) and (index: {}, slice: ({:?}))\nof image \"{}\" in task \"{}\";
                        Intersecting region of ranges: ({:?});
                        All image use ranges must be disjoint in each task!",
                        index_a, range_a,
                        index_b, range_b,
                        self.global_image_infos[handle_index as usize].get_name(),
                        task.get_name(),
                        intersection
                    )
                }
            }
        }

        pub(super) fn create_transient_runtime_buffers(&self, permutation: &mut TaskGraphPermutation) {
            for buffer_info_index in 0..self.global_buffer_infos.len() {
                let global_buffer = &self.global_buffer_infos[buffer_info_index];
                let permutation_buffer = &mut permutation.buffer_infos[buffer_info_index];

                if let PermIndepTaskBufferInfo::Transient { info, memory_requirements } = global_buffer {
                    if permutation_buffer.valid {
                        permutation_buffer.actual_buffer = self.device.create_buffer(BufferInfo {
                            size: info.size,
                            allocation_info: crate::memory_block::AllocationInfo::Manual {
                                memory_block: self.transient_data_memory_block.as_ref().unwrap().clone(),
                                offset: permutation_buffer.allocation_offset
                            },
                            debug_name: info.name.clone().into()
                        }).unwrap()
                    }
                }
            }
        }

        pub(super) fn create_transient_runtime_images(&self, permutation: &mut TaskGraphPermutation) {
            for image_info_index in 0..self.global_image_infos.len() {
                let global_image = &self.global_image_infos[image_info_index];
                let permutation_image = &mut permutation.image_infos[image_info_index];

                if let PermIndepTaskImageInfo::Transient { info, memory_requirements } = global_image {
                    if permutation_image.valid {
                        permutation_image.actual_image = self.device.create_image(ImageInfo {
                            dimensions: info.dimensions,
                            format: info.format,
                            aspect: info.aspect,
                            size: info.size,
                            mip_level_count: info.mip_level_count,
                            array_layer_count: info.array_layer_count,
                            sample_count: info.sample_count,
                            usage: permutation_image.usage,
                            allocation_info: crate::memory_block::AllocationInfo::Manual {
                                memory_block: self.transient_data_memory_block.as_ref().unwrap().clone(),
                                offset: permutation_image.allocation_offset
                            },
                            debug_name: info.name.clone().into()
                        }).unwrap()
                    }
                }
            }
        }

        pub(super) fn allocate_transient_resources(&mut self) {
            // Figure out transient resource sizes
            let mut max_alignment_requirement: ash::vk::DeviceSize = 0;
            for global_image in &mut self.global_image_infos {
                if let PermIndepTaskImageInfo::Transient { info, memory_requirements } = global_image {
                    let image_info = crate::gpu_resources::ImageInfo {
                        dimensions: info.dimensions,
                        format: info.format,
                        aspect: info.aspect,
                        size: info.size,
                        mip_level_count: info.mip_level_count,
                        array_layer_count: info.array_layer_count,
                        sample_count: info.sample_count,
                        usage: ImageUsageFlags::STORAGE,
                        allocation_info: crate::memory_block::AllocationInfo::Automatic(gpu_allocator::MemoryLocation::GpuOnly),
                        debug_name: "Dummy to figure out memory requirements".into()
                    };
                    *memory_requirements = self.device.get_image_memory_requirements(&image_info);
                    max_alignment_requirement = memory_requirements.alignment.max(max_alignment_requirement);
                }
            }
            for global_buffer in &mut self.global_buffer_infos {
                if let PermIndepTaskBufferInfo::Transient { info, memory_requirements } = global_buffer {
                    let buffer_info = crate::gpu_resources::BufferInfo {
                        size: info.size,
                        allocation_info: crate::memory_block::AllocationInfo::Automatic(gpu_allocator::MemoryLocation::GpuOnly),
                        debug_name: "Dummy to figure memory requirements".into()
                    };
                    *memory_requirements = self.device.get_buffer_memory_requirements(&buffer_info);
                    max_alignment_requirement = memory_requirements.alignment.max(max_alignment_requirement);
                }
            }

            // For each permutation, figure out the max memory requirements
            for permutation in &mut self.permutations {
                let mut batches = 0;
                let mut submit_batch_offsets = vec![];
                submit_batch_offsets.resize(permutation.batch_submit_scopes.len(), 0);
                for index in 0..permutation.batch_submit_scopes.len() {
                    submit_batch_offsets[index] = batches;
                    batches += permutation.batch_submit_scopes[index].task_batches.len();
                }

                struct LifetimeLengthResource {
                    start_batch: usize,
                    end_batch: usize,
                    lifetime_length: usize,
                    is_image: bool,
                    resource_index: u32
                }

                let mut lifetime_length_sorted_resources = Vec::<LifetimeLengthResource>::default();

                for permutation_image_index in 0..permutation.image_infos.len() {
                    if let PermIndepTaskImageInfo::Persistent { image } = &self.global_image_infos[permutation_image_index] {
                        continue;
                    }
                    if !permutation.image_infos[permutation_image_index].valid {
                        continue;
                    }

                    let permutation_task_image = &permutation.image_infos[permutation_image_index];

                    if permutation_task_image.lifetime.first_use.submit_scope_index == usize::MAX || permutation_task_image.lifetime.last_use.submit_scope_index == usize::MAX {
                        // TODO(msakmary) Transient image created but not used - should we somehow warn the user about this?
                        permutation.image_infos[permutation_image_index].valid = false;
                        continue;
                    }

                    let start_index = submit_batch_offsets[permutation_task_image.lifetime.first_use.submit_scope_index]
                                        + permutation_task_image.lifetime.first_use.task_batch_index;
                    let end_index = submit_batch_offsets[permutation_task_image.lifetime.last_use.submit_scope_index]
                                        + permutation_task_image.lifetime.last_use.task_batch_index;

                    lifetime_length_sorted_resources.push(LifetimeLengthResource {
                        start_batch: start_index,
                        end_batch: end_index,
                        lifetime_length: end_index - start_index + 1,
                        is_image: true,
                        resource_index: permutation_image_index as u32
                    });
                }

                for permutation_buffer_index in 0..permutation.buffer_infos.len() {
                    if let PermIndepTaskBufferInfo::Persistent { buffer } = &self.global_buffer_infos[permutation_buffer_index] {
                        continue;
                    }

                    let permutation_task_buffer = &permutation.buffer_infos[permutation_buffer_index];
                    
                    if permutation_task_buffer.lifetime.first_use.submit_scope_index == usize::MAX || permutation_task_buffer.lifetime.last_use.submit_scope_index == usize::MAX {
                        // TODO(msakmary) Transient buffer created but not used - should we somehow warn the user about this?
                        permutation.buffer_infos[permutation_buffer_index].valid = false;
                        continue;
                    }
                    
                    let start_index = submit_batch_offsets[permutation_task_buffer.lifetime.first_use.submit_scope_index]
                                        + permutation_task_buffer.lifetime.first_use.task_batch_index;
                    let end_index = submit_batch_offsets[permutation_task_buffer.lifetime.last_use.submit_scope_index]
                                        + permutation_task_buffer.lifetime.last_use.task_batch_index;

                    //debug_assert!(start_index != usize::MAX || end_index != usize::MAX, "Detected transient resource created but never used!");

                    lifetime_length_sorted_resources.push(LifetimeLengthResource {
                        start_batch: start_index,
                        end_batch: end_index,
                        lifetime_length: end_index - start_index + 1,
                        is_image: false,
                        resource_index: permutation_buffer_index as u32
                    });
                }

                lifetime_length_sorted_resources.sort_by(|first, second| {
                    first.lifetime_length.cmp(&second.lifetime_length)
                });

                struct Allocation {
                    offset: usize,
                    size: usize,
                    start_batch: usize,
                    end_batch: usize,
                    is_image: bool,
                    owning_resource_index: u32,
                    memory_type_bits: u32,
                    intersection: ImageSubresourceRange
                }

                // Sort allocations in the set in the following way
                //      1) Sort by offsets into the memory block
                // if equal:
                //      2) Sort by start batch of the allocation
                // if equal:
                //      3) Sort by owning image index
                impl PartialEq for Allocation {
                    fn eq(&self, other: &Self) -> bool {
                        self.offset == other.offset &&
                        self.size == other.size &&
                        self.start_batch == other.start_batch &&
                        self.end_batch == other.end_batch &&
                        self.is_image == other.is_image &&
                        self.owning_resource_index == other.owning_resource_index &&
                        self.memory_type_bits == other.memory_type_bits &&
                        self.intersection.aspect_mask == other.intersection.aspect_mask &&
                        self.intersection.base_mip_level == other.intersection.base_mip_level &&
                        self.intersection.level_count == other.intersection.level_count &&
                        self.intersection.base_array_layer == other.intersection.base_array_layer &&
                        self.intersection.layer_count == other.intersection.layer_count
                    }
                }
                impl Eq for Allocation { }
                impl PartialOrd for Allocation {
                    // (Exsolutus) Implementation is wrong, but never used or exposed so its fine
                    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                        None
                    }
                }
                impl Ord for Allocation {
                    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                        if self.offset < other.offset {
                            return std::cmp::Ordering::Less;
                        }
                        if self.offset == other.offset {
                            if self.start_batch < other.start_batch {
                                return std::cmp::Ordering::Less;
                            }
                            if self.start_batch == other.start_batch {
                                return self.owning_resource_index.cmp(&other.owning_resource_index);
                            }
                            // self.offset == other.offset && self.start_batch > other.start_batch
                            return std::cmp::Ordering::Greater;
                        }
                        // self.offset > other.offset
                        return std::cmp::Ordering::Greater;
                    }
                }

                let mut allocations = std::collections::BTreeSet::<Allocation>::new();
                // Figure out where to allocate each resource
                let mut no_alias_back_offset = 0usize;
                for resource_lifetime in lifetime_length_sorted_resources {
                    let memory_requirements = match resource_lifetime.is_image {
                        true => {
                            match &self.global_image_infos[resource_lifetime.resource_index as usize] {
                                PermIndepTaskImageInfo::Transient { info, memory_requirements } => {
                                    memory_requirements
                                },
                                _ => unreachable!()
                            }
                        },
                        false => {
                            match &self.global_buffer_infos[resource_lifetime.resource_index as usize] {
                                PermIndepTaskBufferInfo::Transient { info, memory_requirements } => {
                                    memory_requirements
                                },
                                _ => unreachable!()
                            }
                        }
                    };
                    // Go through all memory block states in which this resource is alive and try to find a spot for it
                    let mut new_allocation = Allocation {
                        offset: 0,
                        size: memory_requirements.size as usize,
                        start_batch: resource_lifetime.start_batch,
                        end_batch: resource_lifetime.end_batch,
                        is_image: resource_lifetime.is_image,
                        owning_resource_index: resource_lifetime.resource_index,
                        memory_type_bits: memory_requirements.memory_type_bits,
                        intersection: ImageSubresourceRange {
                            aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                            base_mip_level: resource_lifetime.start_batch as u32,
                            level_count: resource_lifetime.lifetime_length as u32,
                            base_array_layer: 0,
                            layer_count: memory_requirements.size as u32
                        }
                    };
                    let align = memory_requirements.alignment.max(1) as usize;

                    if self.info.alias_transients {
                        // TODO(msakmary) Fix the intersect functionality so that it is general and does not do hacky stuff like constructing
                        // a mip array slice
                        // Find space in memory and time the new allocation fits into.
                        for allocation in &allocations {
                            if new_allocation.intersection.intersects(allocation.intersection) {
                                // Assign new offset into the memory block
                                // We need to guarantee correct alignment
                                let current_offset = allocation.offset + allocation.size;
                                let aligned_current_offset = (current_offset + align - 1) / (align * align);
                                new_allocation.offset = aligned_current_offset;
                                new_allocation.intersection.base_array_layer = new_allocation.offset as u32;
                            }
                        }
                    } else {
                        let aligned_current_offset = (no_alias_back_offset + align - 1) / (align * align);
                        new_allocation.offset = aligned_current_offset;
                        no_alias_back_offset = new_allocation.offset + new_allocation.size;
                    }
                    allocations.insert(new_allocation);
                }
                // Once we are done with finding space for all the allocations go through all permutation resources and copy over the allocation information
                for allocation in allocations {
                    if allocation.is_image {
                        permutation.image_infos[allocation.owning_resource_index as usize].allocation_offset = allocation.offset;
                    } else {
                        permutation.buffer_infos[allocation.owning_resource_index as usize].allocation_offset = allocation.offset;
                    }
                    // Find the amount of memory this permutation requires
                    self.memory_block_size = self.memory_block_size.max((allocation.offset + allocation.size) as u64);
                    self.memory_type_bits = self.memory_type_bits | allocation.memory_type_bits;
                }
            }
            self.transient_data_memory_block = Some(self.device.create_memory(MemoryBlockInfo {
                requirements: ash::vk::MemoryRequirements {
                    size: self.memory_block_size,
                    alignment: max_alignment_requirement,
                    memory_type_bits: self.memory_type_bits
                },
                location: gpu_allocator::MemoryLocation::GpuOnly,
                name: "TaskGraph Transient Data MemoryBlock"
            }).expect("Transient Data MemoryBlock should be created."));
        }

        pub(super) fn print_task_buffer_to(&self, out: &mut String, indent: &String, permutation: &TaskGraphPermutation, local_id: TaskBufferHandle) {
            let global_buffer = match local_id {
                TaskBufferHandle::Persistent { index } => &self.global_buffer_infos[index as usize],
                TaskBufferHandle::Transient { index, .. } => &self.global_buffer_infos[index as usize]
            };
            let mut persistent_info = String::new();
            if let PermIndepTaskBufferInfo::Persistent { buffer } = global_buffer {
                persistent_info = format!(", persistent index: {}", buffer.unique_index);
            }
            *out += format!("{}Task buffer name: \"{}\", id: ({}){}\n", indent, global_buffer.get_name(), local_id, persistent_info).as_str();
            *out += format!("{}Runtime buffers:\n", indent).as_str();
            
            for child in self.get_actual_buffers(local_id, permutation).iter() {
                let child_info = self.device.info_buffer(*child);
                *out += format!("{}Name: \"{}\", id: ({:?})\n", indent, child_info.debug_name, child).as_str();
            }
            *out += format!("{}--------------------------------\n", indent).as_str();
        }

        pub(super) fn print_task_image_to(&self, out: &mut String, indent: &String, permutation: &TaskGraphPermutation, local_id: TaskImageHandle) {
            let global_image = match local_id {
                TaskImageHandle::Persistent { index, range } => &self.global_image_infos[index as usize],
                TaskImageHandle::Transient { index, .. } => &self.global_image_infos[index as usize]
            };
            let mut persistent_info = String::new();
            if let PermIndepTaskImageInfo::Persistent { image } = global_image {
                persistent_info = format!(", persistent_index: {}", image.unique_index);
            }
            *out += format!("{}Task image name: \"{}\", id ({}){}\n", indent, global_image.get_name(), local_id, persistent_info).as_str();
            *out += format!("{}Runtime images:\n", indent).as_str();
            
            for child in self.get_actual_images(local_id, permutation).iter() {
                let child_info = self.device.info_image(*child);
                *out += format!("{}Name: \"{}\", id: ({:?})\n", indent, child_info.debug_name, child).as_str();
            }
            *out += format!("{}--------------------------------\n", indent).as_str();
        }

        pub(super) fn print_task_barrier_to(&self, out: &mut String, indent: &String, permutation: &TaskGraphPermutation, index: usize, split_barrier: bool) {
            let from_split;
            let barrier = match split_barrier {
                true => {
                    from_split = TaskBarrier::from(&permutation.split_barriers[index]);
                    &from_split
                },
                false => &permutation.barriers[index]
            };
            let memory_barrier = MemoryBarrierInfo {
                src_access: barrier.src_access,
                dst_access: barrier.dst_access
            };
            if let Some(image_id) = barrier.image_id {
                *out += format!("{}Range: ({:?})\n", indent, barrier.range).as_str();
                *out += format!("{}{:?}\n", indent, memory_barrier).as_str();
                *out += format!("{}Layout: ({:?}) -> ({:?})\n", indent, barrier.src_layout, barrier.dst_layout).as_str();
                self.print_task_image_to(out, indent, permutation, image_id);
            } else {
                *out += format!("{}{:?}\n", indent, memory_barrier).as_str();
            }
        }

        pub(super) fn print_task_to(&self, out: &mut String, indent: &String, permutation: &TaskGraphPermutation, task_id: TaskId) {
            let task = &self.tasks[task_id];
            *out += format!("{}Task name: \"{}\", id: {}\n", indent, task.task.get_name(), task_id).as_str();
            *out += format!("{}Task arguments:\n", indent).as_str();
            let resource_uses = task.task.get_resource_uses();
            for buffer in resource_uses.0 {
                let access = task_buffer_access_to_access(&buffer.access);
                *out += format!("{}Buffer argument:\n", indent).as_str();
                *out += format!("{}Access: ({:?})\n", indent, access).as_str();
                self.print_task_buffer_to(out, indent, permutation, buffer.handle);
                *out += format!("{}--------------------------------\n", indent).as_str();
            }
            for image in resource_uses.1 {
                let (layout, access) = task_image_access_to_layout_access(&image.access);
                *out += format!("{}Image argument:\n", indent).as_str();
                *out += format!("{}Access: ({:?})\n", indent, access).as_str();
                *out += format!("{}Layout: {:?}\n", indent, layout).as_str();
                *out += format!("{}Range: {:?}\n", indent, image.handle.range()).as_str();
                self.print_task_image_to(out, indent, permutation, image.handle);
                *out += format!("{}--------------------------------\n", indent).as_str();
            }
        }

        pub(super) fn print_permutation_aliasing_to(&self, out: &mut String, indent: &String, permutation: &TaskGraphPermutation) {
            let mut batches = 0;
            let mut submit_batch_offsets = Vec::<usize>::new();
            submit_batch_offsets.resize(permutation.batch_submit_scopes.len(), 0);
            for submit_scope_idx in 0..permutation.batch_submit_scopes.len() {
                submit_batch_offsets[submit_scope_idx] = batches;
                batches += permutation.batch_submit_scopes[submit_scope_idx].task_batches.len();
            }
            {
                let print_lifetime = |start_idx: usize, end_idx: usize| -> String {
                    let mut out = String::new();
                    for i in 0..batches {
                        if i >=start_idx && i < end_idx {
                            out += format!("{}===", i).as_str();
                        } else if i == end_idx && end_idx != batches - 1 {
                            out += format!("{}---", i).as_str();
                        } else if i != batches - 1 {
                            out += "----";
                        } else {
                            if end_idx == batches - 1 {
                                out += format!("{}", i).as_str();
                            } else {
                                out += "-";
                            }
                        }
                    }
                    out
                };
                *out += format!("{}Resource lifetimes and aliasing:\n", indent).as_str();
                for perm_image_idx in 0..permutation.image_infos.len() {
                    let PermIndepTaskImageInfo::Transient { info, memory_requirements } = &self.global_image_infos[perm_image_idx] else {
                        continue;
                    };
                    if !permutation.image_infos[perm_image_idx].valid {
                        continue;
                    }

                    let perm_task_image = &permutation.image_infos[perm_image_idx];
                    let start_idx = submit_batch_offsets[perm_task_image.lifetime.first_use.submit_scope_index]
                                    + perm_task_image.lifetime.first_use.task_batch_index;
                    let end_idx = submit_batch_offsets[perm_task_image.lifetime.last_use.submit_scope_index]
                                    + perm_task_image.lifetime.last_use.task_batch_index;
                    *out += format!("{}{}", indent, print_lifetime(start_idx, end_idx)).as_str();
                    *out += format!("{}Allocation offset: {} Allocation size: {} Task resource name: {}\n",
                        indent,
                        perm_task_image.allocation_offset,
                        memory_requirements.size,
                        info.name
                    ).as_str();
                }
                for perm_buffer_idx in 0..permutation.buffer_infos.len() {
                    let PermIndepTaskBufferInfo::Transient { info, memory_requirements } = &self.global_buffer_infos[perm_buffer_idx] else {
                        continue;
                    };
                    if !permutation.buffer_infos[perm_buffer_idx].valid {
                        continue;
                    }

                    let perm_task_buffer = &permutation.buffer_infos[perm_buffer_idx];
                    let start_idx = submit_batch_offsets[perm_task_buffer.lifetime.first_use.submit_scope_index]
                                    + perm_task_buffer.lifetime.first_use.task_batch_index;
                    let end_idx = submit_batch_offsets[perm_task_buffer.lifetime.last_use.submit_scope_index]
                                    + perm_task_buffer.lifetime.last_use.task_batch_index;
                    *out += format!("{}{}", indent, print_lifetime(start_idx, end_idx)).as_str();
                    *out += format!("{}Allocation offset: {} Allocation size: {} Task resource name: {}\n",
                        indent,
                        perm_task_buffer.allocation_offset,
                        memory_requirements.size,
                        info.name
                    ).as_str();
                }
            }
        }

        pub(super) fn debug_print(&mut self) {
            let mut out = String::new();
            out += format!("TaskGraph name: {}, id: {}:\n", self.info.debug_name, self.unique_index).as_str();
            out += format!("Device: {}\n", self.device.info().debug_name).as_str();
            out += format!("Swapchain: {}\n", match &self.info.swapchain { 
                Some(swapchain) => &swapchain.info().debug_name, 
                None => "-"
            }).as_str();
            out += format!("Reorder Tasks: {}\n", self.info.reorder_tasks).as_str();
            out += format!("Use split barriers: {}\n", self.info.use_split_barriers).as_str();
            out += format!("Permutation condition count: {}\n", self.info.permutation_condition_count).as_str();
            out += format!("Enable command labels: {}\n", self.info.enable_command_lables).as_str();
            out += format!("TaskGraph label color: ({}, {}, {}, {})\n",
                self.info.task_graph_label_color[0],
                self.info.task_graph_label_color[1],
                self.info.task_graph_label_color[2],
                self.info.task_graph_label_color[3],
            ).as_str();
            out += format!("Task batch label color: ({}, {}, {}, {})\n",
                self.info.task_batch_label_color[0],
                self.info.task_batch_label_color[1],
                self.info.task_batch_label_color[2],
                self.info.task_batch_label_color[3]
            ).as_str();
            out += format!("Task label color: ({}, {}, {}, {})\n",
                self.info.task_label_color[0],
                self.info.task_label_color[1],
                self.info.task_label_color[2],
                self.info.task_label_color[3]
            ).as_str();
            out += format!("Record debug information: {}\n", self.info.record_debug_information).as_str();
            out += format!("Staging memory pool size: {}\n", self.info.staging_memory_pool_size).as_str();
            out += format!("Executed permutation: {}\n", self.chosen_permutation_last_execution).as_str();
            
            let permutation = &self.permutations[self.chosen_permutation_last_execution as usize];
            {
                self.print_permutation_aliasing_to(&mut out, &"\t".to_string(), permutation);
                
                out += format!("Permutation split barriers: {}\n", self.info.use_split_barriers).as_str();
                for (submit_scope_index, submit_scope) in permutation.batch_submit_scopes.iter().enumerate() {
                    out += format!("\tSubmit scope: {}\n", submit_scope_index).as_str();
                    for (batch_index, batch) in submit_scope.task_batches.iter().enumerate() {
                        out += format!("\t\tBatch: {}\n", batch_index).as_str();
                        out += "\t\tInserted pipeline barriers:\n";
                        for barrier_index in &batch.pipeline_barrier_indices {
                            self.print_task_barrier_to(&mut out, &"\t\t\t".to_string(), permutation, *barrier_index, false);
                            out += "\t\t\t--------------------------------\n";
                        }
                        if !self.info.use_split_barriers {
                            out += "\t\tInserted pipeline barriers (converted from split barriers):\n";
                            for barrier_index in &batch.wait_split_barrier_indices {
                                self.print_task_barrier_to(&mut out, &"\t\t\t".to_string(), permutation, *barrier_index, true);
                                out += "\t\t\t--------------------------------\n";
                            }
                        } else {
                            out += "\t\tInserted split pipeline barrier waits:\n";
                            for barrier_index in &batch.wait_split_barrier_indices {
                                self.print_task_barrier_to(&mut out, &"\t\t\t".to_string(), permutation, *barrier_index, true);
                                out += "\t\t\t--------------------------------\n";
                            }
                        }
                        out += "\t\tTasks:\n";
                        for task_id in &batch.tasks {
                            self.print_task_to(&mut out, &"\t\t\t".to_string(), permutation, *task_id);
                            out += "\t\t\t--------------------------------\n";
                        }
                        if self.info.use_split_barriers {
                            out += "\t\tInserted split barrier signals:\n";
                            for barrier_index in &batch.signal_split_barrier_indices {
                                self.print_task_barrier_to(&mut out, &"\t\t\t".to_string(), permutation, *barrier_index, true);
                                out += "\t\t\t--------------------------------\n";
                            }
                        }
                        out += "\t\t\t--------------------------------\n";
                    }
                    if !submit_scope.last_minute_barrier_indices.is_empty() {
                        out += "\t\tInserted last minute pipeline barriers:\n";
                        for barrier_index in &submit_scope.last_minute_barrier_indices {
                            self.print_task_barrier_to(&mut out, &"\t\t\t".to_string(), permutation, *barrier_index, false);
                            out += "\t\t\t--------------------------------\n";
                        }
                    }
                    if submit_scope_index != permutation.batch_submit_scopes.len() - 1 {
                        out += "\t\t\t -- inserted submit -- \n";
                        if submit_scope.present_semaphores.is_some() {
                            out += "\t\t\t -- inserted present -- \n";
                        }
                    }
                    out += "\t\t\t--------------------------------\n";
                }
                out += "\t\t\t--------------------------------\n";
            }
            self.debug_string_stream += &out;
        }
    }

    #[derive(Default)]
    pub(super) struct TaskRuntimeInterface<'a> {
        //pub task_graph: Option<&'a TaskGraph>,
        pub permutation: Option<&'a TaskGraphPermutation>,
        pub current_task: Option<TaskId>,
        pub constant_buffer_info: Option<crate::command_list::ConstantBufferInfo>,
        pub device_address: ash::vk::DeviceAddress,
        pub reuse_last_command_list: bool,
        pub command_lists: Vec<CommandList>,
        pub last_submit_semaphore: Option<BinarySemaphore>
    }

    fn update_image_initial_access_ranges(task_image: &mut PerPermTaskImage, new_access_range: &mut ExtendedImageRangeState) {
        let mut new_access_ranges: Vec<ImageSubresourceRange> = vec![];

        // We need to test if a new use adds and or subtracts from initial uses.
        // To do that, we need to test if the new access range is accessing a subresource BEFORE all other already stored initial access ranges.
        // We compare all new use ranges with already tracked first uses.
        // We intersect the earlier and later happening ranges with the new range and store the intersection rest or the respective later executing range access.
        // This list will contain the remainder of the new access ranges after intersections.
        new_access_ranges.push(new_access_range.state.range);
        // Name shortening
        let initial_accesses = &mut task_image.first_range_states;
        // Note(pahrens):
        // NEVER access new_access_range.range in this function past this point.
        // ALWAYS access the new access ImageSubresourceRange from an iterator or via vector indexing.
        let mut new_access_range_index = 0;
        while new_access_range_index < new_access_ranges.len() {
            let mut broke_inner_loop = false;
            for mut initial_access_index in 0..initial_accesses.len() {
                let ranges_disjoint = !new_access_ranges[new_access_range_index].intersects(initial_accesses[initial_access_index].state.range);
                let same_batch = new_access_range.latest_access_submit_scope_index == initial_accesses[initial_access_index].latest_access_submit_scope_index
                                    && new_access_range.latest_access_batch_index == initial_accesses[initial_access_index].latest_access_batch_index;
                // We check if the sets are disjoint.
                // If they are we do not need to do anything and advance to the next test.
                // When two accesses are in the same batch and scope, they can not overlap.
                // This is simply forbidden by task list rules!
                if same_batch || ranges_disjoint { continue; }
                // Now that we have this edge case out the way, we now need to test which tracked range is executed earlier.
                let new_use_executes_earlier = new_access_range.latest_access_submit_scope_index < initial_accesses[initial_access_index].latest_access_submit_scope_index
                                                || (new_access_range.latest_access_submit_scope_index == initial_accesses[initial_access_index].latest_access_submit_scope_index 
                                                && new_access_range.latest_access_batch_index < initial_accesses[initial_access_index].latest_access_batch_index);
                // When the new use is executing earlier, we subtract from the current initial access range.
                // We then replace the current initial access range with the resulting rest.
                if new_use_executes_earlier {
                    // When we intersect, we remove the old initial access range and replace it with the rest of the subtraction.
                    // We need a copy of this, as we will erase this value from the vector first.
                    let initial_access_range = initial_accesses.remove(initial_access_index);
                    // Subtract ranges.
                    let (range_rest, range_rest_count) = initial_access_range.state.range.subtract(new_access_ranges[initial_access_index]);
                    // Now construct new sub-ranges from the rest of the subtraction.
                    // We advance the iterator each time.
                    for rest_index in 0..range_rest_count {
                        let mut rest_tracked_range = initial_access_range;
                        rest_tracked_range.state.range = range_rest[rest_index];
                        // We insert into the beginning, so we don't recheck these with the current new use slice.
                        // They are the result of a subtraction, therefore disjoint.
                        initial_accesses.insert(0, rest_tracked_range);
                    }
                    // We erased, so we implicitely advance by an element, as erase moves all elements one to the left past the iterator.
                    // But as we inserted to the front, we need to move the index accordingly to "stay in place".
                    initial_access_index += range_rest_count;
                }
                // When the new use is executing AFTER the current initial access range, we subtract the current initial access range from the enw range.
                // We then replace the current new access range with the resulting rest.
                else {
                    // We subtract the initial use from the new use and append the rest.
                    let (range_rest, range_rest_count) = new_access_ranges[new_access_range_index].subtract(initial_accesses[initial_access_index].state.range);
                    // We insert the rest of the subtraction into the new use list.
                    let mut range_rest_vec = range_rest[0..range_rest_count].to_vec();
                    new_access_ranges.append(&mut range_rest_vec);
                    // We remove the current new use range, as it intersects with an initial use range and is later in the range.
                    new_access_ranges.remove(new_access_range_index);
                    // If we advance the new use index, we restart the inner loop over the initial accesses.
                    broke_inner_loop = true;
                    break;
                }
            }
            // When we broke out the inner loop, we want to "restart" the iteration of the outer loop at the current index.
            if !broke_inner_loop {
                new_access_range_index += 1;
            }
        }
        // Add the newly found initial access ranges to the list of initial access ranges.
        for new_range in new_access_ranges {
            let mut new_tracked_range = new_access_range.clone();
            new_tracked_range.state.range = new_range;
            initial_accesses.push(new_tracked_range);
        }
    }

    fn task_buffer_access_to_access(access: &TaskBufferAccess) -> Access {
        use ash::vk::PipelineStageFlags2 as stage;
        use ash::vk::AccessFlags2 as access;

        const ACCESS_READ_WRITE: access = access::from_raw(0b1000_0000_0000_0000 | 0b1_0000_0000_0000_0000);

        match access {
            TaskBufferAccess::None => crate::types::access_consts::NONE,
            TaskBufferAccess::ShaderRead => Access(stage::ALL_GRAPHICS | stage::COMPUTE_SHADER, access::MEMORY_READ),
            TaskBufferAccess::VertexShaderRead => Access(stage::VERTEX_SHADER, access::MEMORY_READ),
            TaskBufferAccess::TessellationControlShaderRead => Access(stage::TESSELLATION_CONTROL_SHADER, access::MEMORY_READ),
            TaskBufferAccess::TessellationEvaluationShaderRead => Access(stage::TESSELLATION_EVALUATION_SHADER, access::MEMORY_READ),
            TaskBufferAccess::GeometryShaderRead => Access(stage::GEOMETRY_SHADER, access::MEMORY_READ),
            TaskBufferAccess::FragmentShaderRead => Access(stage::FRAGMENT_SHADER, access::MEMORY_READ),
            TaskBufferAccess::ComputeShaderRead => Access(stage::COMPUTE_SHADER, access::MEMORY_READ),
            TaskBufferAccess::ShaderWrite => Access(stage::ALL_GRAPHICS | stage::COMPUTE_SHADER, access::MEMORY_WRITE),
            TaskBufferAccess::VertexShaderWrite => Access(stage::VERTEX_SHADER, access::MEMORY_WRITE),
            TaskBufferAccess::TessellationControlShaderWrite => Access(stage::TESSELLATION_CONTROL_SHADER, access::MEMORY_WRITE),
            TaskBufferAccess::TessellationEvaluationShaderWrite => Access(stage::TESSELLATION_EVALUATION_SHADER, access::MEMORY_WRITE),
            TaskBufferAccess::GeometryShaderWrite => Access(stage::GEOMETRY_SHADER, access::MEMORY_WRITE),
            TaskBufferAccess::FragmentShaderWrite => Access(stage::FRAGMENT_SHADER, access::MEMORY_WRITE),
            TaskBufferAccess::ComputeShaderWrite => Access(stage::COMPUTE_SHADER, access::MEMORY_WRITE),
            TaskBufferAccess::ShaderReadWrite => Access(stage::ALL_GRAPHICS | stage::COMPUTE_SHADER, ACCESS_READ_WRITE),
            TaskBufferAccess::VertexShaderReadWrite => Access(stage::VERTEX_SHADER, ACCESS_READ_WRITE),
            TaskBufferAccess::TessellationControlShaderReadWrite => Access(stage::TESSELLATION_CONTROL_SHADER, ACCESS_READ_WRITE),
            TaskBufferAccess::TessellationEvaluationShaderReadWrite => Access(stage::TESSELLATION_EVALUATION_SHADER, ACCESS_READ_WRITE),
            TaskBufferAccess::GeometryShaderReadWrite => Access(stage::GEOMETRY_SHADER, ACCESS_READ_WRITE),
            TaskBufferAccess::FragmentShaderReadWrite => Access(stage::FRAGMENT_SHADER, ACCESS_READ_WRITE),
            TaskBufferAccess::ComputeShaderReadWrite => Access(stage::COMPUTE_SHADER, ACCESS_READ_WRITE),
            TaskBufferAccess::IndexRead => Access(stage::INDEX_INPUT, access::MEMORY_READ),
            TaskBufferAccess::DrawIndirectInfoRead => Access(stage::DRAW_INDIRECT, access::MEMORY_READ),
            TaskBufferAccess::TransferRead => Access(stage::TRANSFER, access::MEMORY_READ),
            TaskBufferAccess::TransferWrite => Access(stage::TRANSFER, access::MEMORY_WRITE),
            TaskBufferAccess::HostTransferRead => Access(stage::HOST, access::MEMORY_READ),
            TaskBufferAccess::HostTransferWrite => Access(stage::TRANSFER, access::MEMORY_WRITE),
        }
    }

    fn task_image_access_to_layout_access(access: &TaskImageAccess) -> (ImageLayout, Access) {
        use ash::vk::PipelineStageFlags2 as stage;
        use ash::vk::AccessFlags2 as access;

        const ACCESS_READ_WRITE: access = access::from_raw(0b1000_0000_0000_0000 | 0b1_0000_0000_0000_0000);

        match access {
            TaskImageAccess::None => (ImageLayout::UNDEFINED, crate::types::access_consts::NONE),
            TaskImageAccess::ShaderRead => (ImageLayout::READ_ONLY_OPTIMAL, Access(stage::ALL_GRAPHICS | stage::COMPUTE_SHADER, access::MEMORY_READ)),
            TaskImageAccess::VertexShaderRead => (ImageLayout::READ_ONLY_OPTIMAL, Access(stage::VERTEX_SHADER, access::MEMORY_READ)),
            TaskImageAccess::TessellationControlShaderRead => (ImageLayout::READ_ONLY_OPTIMAL, Access(stage::TESSELLATION_CONTROL_SHADER, access::MEMORY_READ)),
            TaskImageAccess::TessellationEvaluationShaderRead => (ImageLayout::READ_ONLY_OPTIMAL, Access(stage::TESSELLATION_EVALUATION_SHADER, access::MEMORY_READ)),
            TaskImageAccess::GeometryShaderRead => (ImageLayout::READ_ONLY_OPTIMAL, Access(stage::GEOMETRY_SHADER, access::MEMORY_READ)),
            TaskImageAccess::FragmentShaderRead => (ImageLayout::READ_ONLY_OPTIMAL, Access(stage::FRAGMENT_SHADER, access::MEMORY_READ)),
            TaskImageAccess::ComputeShaderRead => (ImageLayout::READ_ONLY_OPTIMAL, Access(stage::COMPUTE_SHADER, access::MEMORY_READ)),
            TaskImageAccess::ShaderWrite => (ImageLayout::GENERAL, Access(stage::ALL_GRAPHICS | stage::COMPUTE_SHADER, access::MEMORY_WRITE)),
            TaskImageAccess::VertexShaderWrite => (ImageLayout::GENERAL, Access(stage::VERTEX_SHADER, access::MEMORY_WRITE)),
            TaskImageAccess::TessellationControlShaderWrite => (ImageLayout::GENERAL, Access(stage::TESSELLATION_CONTROL_SHADER, access::MEMORY_WRITE)),
            TaskImageAccess::TessellationEvaluationShaderWrite => (ImageLayout::GENERAL, Access(stage::TESSELLATION_EVALUATION_SHADER, access::MEMORY_WRITE)),
            TaskImageAccess::GeometryShaderWrite => (ImageLayout::GENERAL, Access(stage::GEOMETRY_SHADER, access::MEMORY_WRITE)),
            TaskImageAccess::FragmentShaderWrite => (ImageLayout::GENERAL, Access(stage::FRAGMENT_SHADER, access::MEMORY_WRITE)),
            TaskImageAccess::ComputeShaderWrite => (ImageLayout::GENERAL, Access(stage::COMPUTE_SHADER, access::MEMORY_WRITE)),
            TaskImageAccess::ShaderReadWrite => (ImageLayout::GENERAL, Access(stage::ALL_GRAPHICS | stage::COMPUTE_SHADER, ACCESS_READ_WRITE)),
            TaskImageAccess::VertexShaderReadWrite => (ImageLayout::GENERAL, Access(stage::VERTEX_SHADER, ACCESS_READ_WRITE)),
            TaskImageAccess::TessellationControlShaderReadWrite => (ImageLayout::GENERAL, Access(stage::TESSELLATION_CONTROL_SHADER, ACCESS_READ_WRITE)),
            TaskImageAccess::TessellationEvaluationShaderReadWrite => (ImageLayout::GENERAL, Access(stage::TESSELLATION_EVALUATION_SHADER, ACCESS_READ_WRITE)),
            TaskImageAccess::GeometryShaderReadWrite => (ImageLayout::GENERAL, Access(stage::GEOMETRY_SHADER, ACCESS_READ_WRITE)),
            TaskImageAccess::FragmentShaderReadWrite => (ImageLayout::GENERAL, Access(stage::FRAGMENT_SHADER, ACCESS_READ_WRITE)),
            TaskImageAccess::ComputeShaderReadWrite => (ImageLayout::GENERAL, Access(stage::COMPUTE_SHADER, ACCESS_READ_WRITE)),
            TaskImageAccess::TransferRead => (ImageLayout::TRANSFER_SRC_OPTIMAL, Access(stage::TRANSFER, access::MEMORY_READ)),
            TaskImageAccess::TransferWrite => (ImageLayout::TRANSFER_DST_OPTIMAL, Access(stage::TRANSFER, access::MEMORY_WRITE)),
            TaskImageAccess::ColorAttachment => (ImageLayout::ATTACHMENT_OPTIMAL, Access(stage::COLOR_ATTACHMENT_OUTPUT, ACCESS_READ_WRITE)),
            TaskImageAccess::DepthAttachment |
            TaskImageAccess::StencilAttachment |
            TaskImageAccess::DepthStencilAttachment => (ImageLayout::ATTACHMENT_OPTIMAL, Access(stage::EARLY_FRAGMENT_TESTS | stage::LATE_FRAGMENT_TESTS, ACCESS_READ_WRITE)),
            TaskImageAccess::DepthAttachmentRead |
            TaskImageAccess::StencilAttachmentRead |
            TaskImageAccess::DepthStencilAttachmentRead => (ImageLayout::READ_ONLY_OPTIMAL, Access(stage::EARLY_FRAGMENT_TESTS | stage::LATE_FRAGMENT_TESTS, access::MEMORY_READ)),
            TaskImageAccess::ResolveWrite => (ImageLayout::ATTACHMENT_OPTIMAL, Access(stage::RESOLVE, access::MEMORY_WRITE)),
            TaskImageAccess::Present => (ImageLayout::PRESENT_SRC_KHR, Access(stage::ALL_COMMANDS, access::MEMORY_READ)),
        }
    }

    fn task_image_access_to_usage(access: &TaskImageAccess) -> ImageUsageFlags {
        match access {
            TaskImageAccess::ShaderRead |
            TaskImageAccess::VertexShaderRead |
            TaskImageAccess::TessellationControlShaderRead |
            TaskImageAccess::TessellationEvaluationShaderRead |
            TaskImageAccess::GeometryShaderRead |
            TaskImageAccess::FragmentShaderRead |
            TaskImageAccess::ComputeShaderRead => ImageUsageFlags::SAMPLED,
            TaskImageAccess::ShaderWrite |
            TaskImageAccess::VertexShaderWrite |
            TaskImageAccess::TessellationControlShaderWrite |
            TaskImageAccess::TessellationEvaluationShaderWrite |
            TaskImageAccess::GeometryShaderWrite |
            TaskImageAccess::FragmentShaderWrite |
            TaskImageAccess::ComputeShaderWrite |
            TaskImageAccess::ShaderReadWrite |
            TaskImageAccess::VertexShaderReadWrite |
            TaskImageAccess::TessellationControlShaderReadWrite |
            TaskImageAccess::TessellationEvaluationShaderReadWrite |
            TaskImageAccess::GeometryShaderReadWrite |
            TaskImageAccess::FragmentShaderReadWrite |
            TaskImageAccess::ComputeShaderReadWrite => ImageUsageFlags::STORAGE,
            TaskImageAccess::TransferRead => ImageUsageFlags::TRANSFER_SRC,
            TaskImageAccess::TransferWrite => ImageUsageFlags::TRANSFER_DST,
            // NOTE(msakmary) - not fully sure about the resolve being color attachment usage
            // this is the best I could deduce from vulkan docs
            TaskImageAccess::ResolveWrite |
            TaskImageAccess::ColorAttachment => ImageUsageFlags::COLOR_ATTACHMENT,
            TaskImageAccess::DepthAttachment |
            TaskImageAccess::StencilAttachment |
            TaskImageAccess::DepthStencilAttachment |
            TaskImageAccess::DepthAttachmentRead |
            TaskImageAccess::StencilAttachmentRead |
            TaskImageAccess::DepthStencilAttachmentRead => ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            TaskImageAccess::Present |
            TaskImageAccess::None => ImageUsageFlags::default()
        }
    }
}