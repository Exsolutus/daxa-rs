use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use spirv_std::macros::gpu_only;
//spirv macro
use spirv_std::spirv;


//use spirv_std::num_traits::Float;


#[spirv(typed_buffer)]
pub struct TypedBuffer<T: ?Sized> {
    // spooky! this field does not exist, so if it's referenced in rust code, things will explode
    _do_not_touch: u32,
    _phantom: PhantomData<T>,
}

impl<T> Deref for TypedBuffer<T> {
    type Target = T;
    #[gpu_only]
    fn deref(&self) -> &T {
        unsafe {
            core::arch::asm! {
                "%uint = OpTypeInt 32 0",
                "%uint_0 = OpConstant %uint 0",
                "%inner_ptr = OpAccessChain _ {buffer} %uint_0",
                "OpReturnValue %inner_ptr",
                buffer = in(reg) self,
                options(noreturn),
            }
        }
    }
}

impl<T> DerefMut for TypedBuffer<T> {
    #[gpu_only]
    fn deref_mut(&mut self) -> &mut T {
        unsafe {
            core::arch::asm! {
                "%uint = OpTypeInt 32 0",
                "%uint_0 = OpConstant %uint 0",
                "%inner_ptr = OpAccessChain _ {buffer} %uint_0",
                "OpReturnValue %inner_ptr",
                buffer = in(reg) self,
                options(noreturn),
            }
        }
    }
}

impl<T> Deref for TypedBuffer<[T]> {
    type Target = [T];
    #[gpu_only]
    fn deref(&self) -> &[T] {
        unsafe {
            core::arch::asm! {
                "%uint = OpTypeInt 32 0",
                "%uint_0 = OpConstant %uint 0",
                "%inner_ptr = OpAccessChain _ {buffer} %uint_0",
                "%inner_len = OpArrayLength %uint {buffer} 0",
                "%inner_slice_ptr = OpCompositeConstruct typeof*{dummy_ref_to_slice_ref} %inner_ptr %inner_len",
                "OpReturnValue %inner_slice_ptr",
                buffer = in(reg) self,
                dummy_ref_to_slice_ref = in(reg) &(&[] as &[T]),
                options(noreturn),
            }
        }
    }
}

impl<T> DerefMut for TypedBuffer<[T]> {
    #[gpu_only]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            core::arch::asm! {
                "%uint = OpTypeInt 32 0",
                "%uint_0 = OpConstant %uint 0",
                "%inner_ptr = OpAccessChain _ {buffer} %uint_0",
                "%inner_len = OpArrayLength %uint {buffer} 0",
                "%inner_slice_ptr = OpCompositeConstruct typeof*{dummy_ref_to_slice_ref} %inner_ptr %inner_len",
                "OpReturnValue %inner_slice_ptr",
                buffer = in(reg) self,
                dummy_ref_to_slice_ref = in(reg) &(&[] as &[T]),
                options(noreturn),
            }
        }
    }
}