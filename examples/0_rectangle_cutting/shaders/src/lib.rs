#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]


use rectangle_cutting_shaders_shared::*;

use spirv_std::{
    spirv,
    RuntimeArray,
    TypedBuffer,
    glam::{
        Vec4,
        Vec4Swizzles
    },
};



#[spirv(vertex)]
pub fn main_vs(
    // Input Parameters
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] buffers: &RuntimeArray<TypedBuffer<RuntimeArray<DrawVertex>>>,
    #[spirv(push_constant)] push_constant: &DrawPush,
    #[spirv(vertex_index)] vertex_index: u32,
    // Output Parameters
    #[spirv(position)] out_pos: &mut Vec4,
    out_color: &mut Vec4,
) {
    let vert = unsafe { buffers.index(push_constant.face_buffer as usize).index(vertex_index as usize) };
    *out_pos = Vec4::from((vert.position.xy(), 0.0, 1.0));
    *out_color = vert.color;
}

#[spirv(fragment)]
pub fn main_fs(
    frag_color: Vec4,
    out_color: &mut Vec4
) {
    *out_color = frag_color;
}
