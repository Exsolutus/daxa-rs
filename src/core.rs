use ash::vk;

pub const MAX_PUSH_CONSTANT_WORD_SIZE: u32 = 32;
pub const MAX_PUSH_CONSTANT_BYTE_SIZE: u32 = MAX_PUSH_CONSTANT_WORD_SIZE * 4;
pub const PIPELINE_LAYOUT_COUNT: u32 = MAX_PUSH_CONSTANT_WORD_SIZE + 1;
pub const MAX_PUSH_CONSTANT_SIZE_ERROR: &str = concat!("Push constant size is limited to 128 bytes / 32 device words");

