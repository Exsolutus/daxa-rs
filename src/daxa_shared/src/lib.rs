#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(asm_experimental_arch)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

// #[cfg(target_arch = "spirv")]
