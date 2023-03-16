#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

mod test0;
mod test1;

use test0::test1::func;

use spirv_std::spirv;



// LocalSize/numthreads of (x = 8, y = 8, z = 1)
#[spirv(compute(threads(8, 8, 1)))]
pub fn main_cs(

) {
    func();
}
