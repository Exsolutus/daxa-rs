use ash::vk;

use std::ops::BitOr;

// reexport
pub use {
    vk::Rect2D,
    vk::Offset2D,
    vk::Extent2D,
    vk::Extent3D,
    vk::Viewport,
    vk::ImageSubresourceRange
};


#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Access(pub(crate) vk::PipelineStageFlags2, pub(crate) vk::AccessFlags2);

impl Default for Access {
    fn default() -> Self {
        access_consts::NONE
    }
}

impl BitOr for Access {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0, self.1 | rhs.1)
    }
}

pub mod access_consts {
    use super::Access;
    use ash::vk;

    pub const NONE: Access = Access(vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);

    pub const VERTEX_SHADER_READ: Access = Access(vk::PipelineStageFlags2::VERTEX_SHADER, vk::AccessFlags2::MEMORY_READ);
    pub const TRANSFER_READ: Access = Access(vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::MEMORY_READ);
    pub const HOST_READ: Access = Access(vk::PipelineStageFlags2::HOST, vk::AccessFlags2::MEMORY_READ);

    pub const COLOR_ATTACHMENT_OUTPUT_WRITE: Access = Access(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::MEMORY_WRITE);
    pub const TRANSFER_WRITE: Access = Access(vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::MEMORY_WRITE);
    pub const HOST_WRITE: Access = Access(vk::PipelineStageFlags2::HOST, vk::AccessFlags2::MEMORY_WRITE);
    
    const ACCESS_READ_WRITE: vk::AccessFlags2 = vk::AccessFlags2::from_raw(0b1000_0000_0000_0000 | 0b1_0000_0000_0000_0000);
    pub const TRANSFER_READ_WRITE: Access = Access(vk::PipelineStageFlags2::TRANSFER, ACCESS_READ_WRITE);
    pub const ALL_GRAPHICS_READ_WRITE: Access = Access(vk::PipelineStageFlags2::ALL_GRAPHICS, ACCESS_READ_WRITE);
}
