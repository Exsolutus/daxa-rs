use ash::vk;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct Access(pub(crate) vk::PipelineStageFlags2, pub(crate) vk::AccessFlags2);

impl Default for Access {
    fn default() -> Self {
        access_consts::NONE
    }
}

pub mod access_consts {
    use super::Access;
    use ash::vk;

    pub const NONE: Access = Access(vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);

    pub const TRANSFER_READ: Access = Access(vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::MEMORY_READ);
    pub const HOST_READ: Access = Access(vk::PipelineStageFlags2::HOST, vk::AccessFlags2::MEMORY_READ);

    pub const TRANSFER_WRITE: Access = Access(vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::MEMORY_WRITE);
    pub const HOST_WRITE: Access = Access(vk::PipelineStageFlags2::HOST, vk::AccessFlags2::MEMORY_WRITE);
    
}