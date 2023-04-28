#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use spirv_std::glam;


#[repr(C)]
pub struct DrawVertex {
    pub position: glam::Vec4,
    pub color: glam::Vec4
}

#[repr(C)]
pub struct DrawVertexBuffer {
    pub verts: [DrawVertex; 3]
}

#[repr(C)]
pub struct DrawPush {
    pub face_buffer: u32
}