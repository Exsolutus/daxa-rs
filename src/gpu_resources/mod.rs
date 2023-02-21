use ash::vk;

use std::{
    fmt::Display,
};



#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct GPUResourceId(pub u32);

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct BufferId(pub u32);

impl Display for BufferId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct ImageId(pub u32);

impl Display for ImageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct ImageViewId(pub u32);

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct SamplerId(pub u32);


