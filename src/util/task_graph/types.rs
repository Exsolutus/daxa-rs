use crate::gpu_resources::{
    BufferId,
    ImageId,
    ImageViewId,
    ImageSubresourceRange
};


#[derive(Clone, Copy, Debug, Default)]
pub enum TaskBufferAccess {
    #[default]
    None,
    ShaderRead,
    VertexShaderRead,
    TessellationControlShaderRead,
    TessellationEvaluationShaderRead,
    GeometryShaderRead,
    FragmentShaderRead,
    ComputeShaderRead,
    ShaderWrite,
    VertexShaderWrite,
    TessellationControlShaderWrite,
    TessellationEvaluationShaderWrite,
    GeometryShaderWrite,
    FragmentShaderWrite,
    ComputeShaderWrite,
    ShaderReadWrite,
    VertexShaderReadWrite,
    TessellationControlShaderReadWrite,
    TessellationEvaluationShaderReadWrite,
    GeometryShaderReadWrite,
    FragmentShaderReadWrite,
    ComputeShaderReadWrite,
    IndexRead,
    DrawIndirectInfoRead,
    TransferRead,
    TransferWrite,
    HostTransferRead,
    HostTransferWrite,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum TaskImageAccess {
    #[default]
    None,
    ShaderRead,
    VertexShaderRead,
    TessellationControlShaderRead,
    TessellationEvaluationShaderRead,
    GeometryShaderRead,
    FragmentShaderRead,
    ComputeShaderRead,
    ShaderWrite,
    VertexShaderWrite,
    TessellationControlShaderWrite,
    TessellationEvaluationShaderWrite,
    GeometryShaderWrite,
    FragmentShaderWrite,
    ComputeShaderWrite,
    ShaderReadWrite,
    VertexShaderReadWrite,
    TessellationControlShaderReadWrite,
    TessellationEvaluationShaderReadWrite,
    GeometryShaderReadWrite,
    FragmentShaderReadWrite,
    ComputeShaderReadWrite,
    TransferRead,
    TransferWrite,
    ColorAttachment,
    DepthAttachment,
    StencilAttachment,
    DepthStencilAttachment,
    DepthAttachmentRead,
    StencilAttachmentRead,
    DepthStencilAttachmentRead,
    ResolveWrite,
    Present,
}



type TaskResourceIndex = u32;


pub(crate) trait TaskGPUResourceHandle { }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TaskBufferHandle {
    Persistent {
        index: TaskResourceIndex
    },
    Transient {
        task_graph_index: TaskResourceIndex,
        index: TaskResourceIndex
    }
}
impl TaskGPUResourceHandle for TaskBufferHandle { }
impl std::fmt::Display for TaskBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskBufferHandle::Persistent { index } => write!(f, "index: {}", index),
            TaskBufferHandle::Transient { task_graph_index, index } => write!(f, "task_graph_index: {}, index: {}", task_graph_index, index)
        }
    }
}

impl TaskBufferHandle {
    pub fn index(&self) -> TaskResourceIndex {
        match self {
            TaskBufferHandle::Persistent { index, .. } => *index,
            TaskBufferHandle::Transient { index, ..} => *index
        }
    }
}

#[derive(Clone, Copy)]
pub enum TaskImageHandle {
    Persistent {
        index: TaskResourceIndex,
        range: ImageSubresourceRange
    },
    Transient {
        task_graph_index: TaskResourceIndex,
        index: TaskResourceIndex,
        range: ImageSubresourceRange
    },
}
impl TaskGPUResourceHandle for TaskImageHandle { }
impl std::fmt::Display for TaskImageHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskImageHandle::Persistent { index, range } => write!(f, "index: {}", index),
            TaskImageHandle::Transient { task_graph_index, index, range } => write!(f, "task_graph_index: {}, index: {}", task_graph_index, index)
        }
    }
}
impl PartialEq for TaskImageHandle {
    fn eq(&self, other: &Self) -> bool {
        match self {
            TaskImageHandle::Persistent { index, range } => {
                let first = index;
                match other {
                    TaskImageHandle::Persistent { index, range } => {
                        let second = index;
                        first == second
                    },
                    _ => false
                }
            },
            TaskImageHandle::Transient { task_graph_index, index, range } => {
                let first = index;
                match other {
                    TaskImageHandle::Transient { task_graph_index, index, range } => {
                        let second = index;
                        first == second
                    },
                    _ => false
                }
            },
            _ => false
        }
    }
}
impl Eq for TaskImageHandle { }

impl TaskImageHandle {
    pub fn index(&self) -> TaskResourceIndex {
        match self {
            TaskImageHandle::Persistent { index, .. } => *index,
            TaskImageHandle::Transient { index, ..} => *index
        }
    }

    pub fn range(&self) -> ImageSubresourceRange {
        match self {
            TaskImageHandle::Persistent { range, .. } => *range,
            TaskImageHandle::Transient { range, .. } => *range
        }
    }

    pub fn subrange(&self, new_range: ImageSubresourceRange) -> TaskImageHandle {
        match self {
            TaskImageHandle::Persistent { index, range } => {
                TaskImageHandle::Persistent { index: *index, range: new_range }
            },
            TaskImageHandle::Transient { task_graph_index, index, range } => {
                TaskImageHandle::Transient { 
                    task_graph_index: *task_graph_index, 
                    index: *index, 
                    range: new_range }
            },
        }
    }

    pub fn view(&self, new_range: ImageSubresourceRange) -> TaskImageHandle {
        let mut ret = self.clone();
        match &mut ret {
            TaskImageHandle::Persistent { range, .. } => {
                *range = new_range;
            },
            TaskImageHandle::Transient { range, .. } => {
                *range = new_range
            }
        };
        ret
    }
}

#[derive(Clone, Copy, Default)]
pub struct ImageRangeState {
    pub latest_access: crate::types::Access,
    pub latest_layout: ash::vk::ImageLayout,
    pub range: ImageSubresourceRange
}



pub(crate) trait TaskResourceUse { }

pub struct TaskBufferUse {
        pub handle: TaskBufferHandle,
        pub(crate) access: TaskBufferAccess,
        pub(crate) buffers: Box<[BufferId]>,
}
impl TaskResourceUse for TaskBufferUse { }

impl TaskBufferUse {
    pub fn new(handle: TaskBufferHandle, access: TaskBufferAccess) -> Self {
        Self {
            handle,
            access,
            buffers: Default::default()
        }
    }

    pub fn access(&self) -> TaskBufferAccess {
        self.access
    }

    pub fn buffer(&self, index: u32) -> BufferId {
        debug_assert!(self.buffers.len() > 0, "This function should only be called within a task callback.");

        self.buffers[index as usize]
    }
}


pub struct TaskImageUse {
    pub handle: TaskImageHandle,
    pub(crate) access: TaskImageAccess,
    pub(crate) view_type: ash::vk::ImageViewType,
    pub(crate) images: Box<[ImageId]>,
    pub(crate) views: Box<[ImageViewId]>
}
impl TaskResourceUse for TaskImageUse { }

impl TaskImageUse {
    pub fn new(handle: TaskImageHandle, access: TaskImageAccess, view_type: ash::vk::ImageViewType) -> Self {
        Self {
            handle,
            access,
            view_type,
            images: Default::default(),
            views: Default::default()
        }
    }

    pub fn image(&self, index: u32) -> ImageId {
        debug_assert!(self.images.len() > 0, "This function should only be called within a task callback.");

        self.images[index as usize]
    }

    pub fn view(&self, index: u32) -> ImageViewId {
        debug_assert!(self.views.len() > 0, "This function should only be called within a task callback.");

        self.views[index as usize]
    }
}



pub(crate) trait BaseTask {
    fn get_resource_uses(&self) -> (&[TaskBufferUse], &[TaskImageUse]);
    fn get_resource_uses_mut(&mut self) -> (&mut [TaskBufferUse], &mut [TaskImageUse]);
    //fn get_uses_constant_buffer_slot(&self) -> isize;
    fn get_name(&self) -> String;
    fn callback(&self, interface: &super::TaskInterface);
}

pub(crate) trait UserTask {

}

pub(crate) struct PredeclaredTask<T: UserTask> {
    task: T,

}

pub(crate) struct InlineTask {
    pub uses: (Box<[TaskBufferUse]>, Box<[TaskImageUse]>),
    pub callback_lambda: super::TaskCallback,
    pub name: String,
    pub constant_buffer_slot: isize
}

impl Default for InlineTask {
    fn default() -> Self {
        Self {
            uses: Default::default(),
            callback_lambda: |_| {},
            name: "".into(),
            constant_buffer_slot: -1
        }
    }
}

impl BaseTask for InlineTask {
    fn get_resource_uses(&self) -> (&[TaskBufferUse], &[TaskImageUse]) {
        (&self.uses.0, &self.uses.1)
    }

    fn get_resource_uses_mut(&mut self) -> (&mut [TaskBufferUse], &mut [TaskImageUse]) {
        (&mut self.uses.0, &mut self.uses.1)
    }

    // fn get_uses_constant_buffer_slot(&self) -> isize {
    //     self.constant_buffer_slot
    // }

    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn callback(&self, interface: &super::TaskInterface) {
        (self.callback_lambda)(interface);
    }
}