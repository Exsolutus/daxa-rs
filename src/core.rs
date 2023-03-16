pub use daxa_derive::{
    ResourceId,
    Slot
};

use ash::vk;



// Constants
pub const MAX_PUSH_CONSTANT_WORD_SIZE: u32 = 32;
pub const MAX_PUSH_CONSTANT_BYTE_SIZE: u32 = MAX_PUSH_CONSTANT_WORD_SIZE * 4;
pub const PIPELINE_LAYOUT_COUNT: u32 = MAX_PUSH_CONSTANT_WORD_SIZE + 1;
pub const MAX_PUSH_CONSTANT_SIZE_ERROR: &str = concat!("Push constant size is limited to 128 bytes / 32 device words");



// Types



// Traits
pub trait ResourceId {
    fn is_empty(&self) -> bool;

    fn index(&self) -> u32;
    fn set_index(&mut self, index: u32);
    fn version(&self) -> u8;
    fn set_version(&mut self, index: u8);
}

pub(crate) trait Slot {
    fn is_zombie(&self) -> bool;
}

pub trait Set {
    fn contains(&self, other: vk::ImageSubresourceRange) -> bool;
    fn intersects(&self, other: vk::ImageSubresourceRange) -> bool;
    fn intersect(&self, other: vk::ImageSubresourceRange) -> vk::ImageSubresourceRange;
    fn subtract(&self, other: vk::ImageSubresourceRange) -> ([vk::ImageSubresourceRange; 4], usize);
}

impl Set for vk::ImageSubresourceRange {
    #[inline]
    fn contains(&self, other: vk::ImageSubresourceRange) -> bool {
        let a_mip_p0 = self.base_mip_level;
        let a_mip_p1 = self.base_mip_level + self.level_count - 1;
        let b_mip_p0 = other.base_mip_level;
        let b_mip_p1 = other.base_mip_level + other.level_count - 1;

        let a_arr_p0 = self.base_array_layer;
        let a_arr_p1 = self.base_array_layer + self.layer_count - 1;
        let b_arr_p0 = other.base_array_layer;
        let b_arr_p1 = other.base_array_layer + other.layer_count - 1;

        b_mip_p0 >= a_mip_p0 &&
        b_mip_p1 <= a_mip_p1 &&
        b_arr_p0 >= a_arr_p0 &&
        b_arr_p1 <= a_arr_p1 &&
        self.aspect_mask == other.aspect_mask
    }

    #[inline]
    fn intersects(&self, other: vk::ImageSubresourceRange) -> bool {
        let a_mip_p0 = self.base_mip_level;
        let a_mip_p1 = self.base_mip_level + self.level_count - 1;
        let b_mip_p0 = other.base_mip_level;
        let b_mip_p1 = other.base_mip_level + other.level_count - 1;

        let a_arr_p0 = self.base_array_layer;
        let a_arr_p1 = self.base_array_layer + self.layer_count - 1;
        let b_arr_p0 = other.base_array_layer;
        let b_arr_p1 = other.base_array_layer + other.layer_count - 1;

        let mip_disjoint = (a_mip_p1 < b_mip_p0) || (b_mip_p1 < a_mip_p0);
        let arr_disjoint = (a_arr_p1 < b_arr_p0) || (b_arr_p1 < a_arr_p0);
        let aspect_disjoint = !((self.aspect_mask & other.aspect_mask) != vk::ImageAspectFlags::NONE);

        !mip_disjoint && !arr_disjoint && !aspect_disjoint
    }

    #[inline]
    fn intersect(&self, other: vk::ImageSubresourceRange) -> vk::ImageSubresourceRange {
        let a_mip_p0 = self.base_mip_level;
        let a_mip_p1 = self.base_mip_level + self.level_count - 1;
        let b_mip_p0 = other.base_mip_level;
        let b_mip_p1 = other.base_mip_level + other.level_count - 1;
        let max_mip_p0 = a_mip_p0.max(b_mip_p0);
        let min_mip_p1 = a_mip_p1.min(b_mip_p1);

        let a_arr_p0 = self.base_array_layer;
        let a_arr_p1 = self.base_array_layer + self.layer_count - 1;
        let b_arr_p0 = other.base_array_layer;
        let b_arr_p1 = other.base_array_layer + other.layer_count - 1;
        let max_arr_p0 = a_arr_p0.max(b_arr_p0);
        let min_arr_p1 = a_arr_p1.min(b_arr_p1);

        // NOTE(grundlett): This multiplication at the end is to cancel out
        // the potential underflow of unsigned integers. Since the p1 could
        // could technically be less than the p0, this means that after doing
        // p1 + 1 - p0, you should get a "negative" number.
        let mip_n = (min_mip_p1 + 1 - max_mip_p0) * ((max_mip_p0 <= min_mip_p1) as u32);
        let arr_n = (min_arr_p1 + 1 - max_arr_p0) * ((max_arr_p0 <= min_arr_p1) as u32);

        vk::ImageSubresourceRange::builder()
            .aspect_mask(self.aspect_mask & other.aspect_mask)
            .base_mip_level(max_mip_p0)
            .level_count(mip_n)
            .base_array_layer(max_arr_p0)
            .layer_count(arr_n)
            .build()
    }

    fn subtract(&self, other: vk::ImageSubresourceRange) -> ([vk::ImageSubresourceRange; 4], usize) {
        let a_mip_p0 = self.base_mip_level;
        let a_mip_p1 = self.base_mip_level + self.level_count - 1;
        let b_mip_p0 = other.base_mip_level;
        let b_mip_p1 = other.base_mip_level + other.level_count - 1;

        let a_arr_p0 = self.base_array_layer;
        let a_arr_p1 = self.base_array_layer + self.layer_count - 1;
        let b_arr_p0 = other.base_array_layer;
        let b_arr_p1 = other.base_array_layer + other.layer_count - 1;

        let mip_case = ((b_mip_p1 < a_mip_p1) as u32) + ((b_mip_p0 > a_mip_p0) as u32) * 2;
        let arr_case = ((b_arr_p1 < a_arr_p1) as u32) + ((b_arr_p0 > a_arr_p0) as u32) * 2;

        let mut result = ([vk::ImageSubresourceRange::default(); 4], 0usize);
        if !self.intersects(other) {
            result.1 = 1;
            result.0[0] = *self;

            // TODO(grundlett): [aspect] If we want to do aspect cutting, we can
            // but we would need to look into it more.
            // result_rects[0].image_aspect &= ~slice.image_aspect;

            return result;
        };

        //
        //     mips ➡️
        // arrays       0              1          2            3
        //  ⬇️
        //
        //           ▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓             ▓▓▓▓       ▓▓  
        //  0      A ▓▓██████▓▓   B ▓▓██░░░░   C ░░░░██▓▓   D ░░██░░
        //           ▓▓██████▓▓     ▓▓██░░░░     ░░░░██▓▓     ░░██░░
        //           ▓▓██████▓▓     ▓▓██░░░░     ░░░░██▓▓     ░░██░░
        //           ▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓             ▓▓▓▓       ▓▓  
        //
        //           ▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓             ▓▓▓▓       ▓▓  
        //  1      E ▓▓██████▓▓   F ▓▓██░░░░   G ░░░░██▓▓   H ░░██░░
        //             ░░░░░░         ░░░░░░     ░░░░░░       ░░░░░░
        //             ░░░░░░         ░░░░░░     ░░░░░░       ░░░░░░
        //
        //  3      I   ░░░░░░     J   ░░░░░░   K ░░░░░░     L ░░░░░░
        //             ░░░░░░         ░░░░░░     ░░░░░░       ░░░░░░
        //           ▓▓██████▓▓     ▓▓██░░░░     ░░░░██▓▓     ░░██░░
        //           ▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓             ▓▓▓▓       ▓▓  
        //
        //  2      M   ░░░░░░     N   ░░░░░░   O ░░░░░░     P ░░░░░░
        //           ▓▓██████▓▓     ▓▓██░░░░     ░░░░██▓▓     ░░██░░
        //             ░░░░░░         ░░░░░░     ░░░░░░       ░░░░░░
        //

        const rect_n: [usize; 16] = [
            0, 1, 1, 2,
            1, 2, 2, 3,
            1, 2, 2, 3,
            2, 3, 3, 4,
        ];

        type RectBCIndices = (usize, usize);

        const NO_RBC: RectBCIndices = (0, 0);

        //   0      1      2      3      4      5
        // b1>a1  a0>b0  a0>a1  a0>b1  b0>b1  b0>a1
        const bc_indices: [[RectBCIndices; 4]; 16] = [
            [NO_RBC, NO_RBC, NO_RBC, NO_RBC],   [(0, 2), NO_RBC, NO_RBC, NO_RBC],   [(1, 2), NO_RBC, NO_RBC, NO_RBC],   [(1, 2), (0, 2), NO_RBC, NO_RBC],
            [(2, 0), NO_RBC, NO_RBC, NO_RBC],   [(0, 3), (2, 0), NO_RBC, NO_RBC],   [(1, 3), (2, 0), NO_RBC, NO_RBC],   [(1, 3), (0, 3), (2, 0), NO_RBC],
            [(2, 1), NO_RBC, NO_RBC, NO_RBC],   [(2, 1), (0, 5), NO_RBC, NO_RBC],   [(2, 1), (1, 5), NO_RBC, NO_RBC],   [(2, 1), (1, 5), (0, 5), NO_RBC],
            [(2, 1), (2, 0), NO_RBC, NO_RBC],   [(2, 1), (0, 4), (2, 0), NO_RBC],   [(2, 1), (1, 4), (2, 0), NO_RBC],   [(2, 1), (1, 4), (0, 4), (2, 0)],
        ];

        struct BaseAndCount {
            base: u32,
            count: u32
        }

        let mip_bc = [
            BaseAndCount { base: b_mip_p1 + 1, count: (a_mip_p1 + 1) - (b_mip_p1 + 1) },    // b1 -> a1
            BaseAndCount { base: a_mip_p0, count: b_mip_p0 - a_mip_p0 },                    // a0 -> b0
            BaseAndCount { base: a_mip_p0, count: (a_mip_p1 + 1) - a_mip_p0 },              // a0 -> a1
        ];
        let arr_bc = [
            BaseAndCount { base: b_arr_p1 + 1, count: (a_arr_p1 + 1) - (b_arr_p1 + 1) },    // b1 -> a1
            BaseAndCount { base: a_arr_p0, count: b_arr_p0 - a_arr_p0 },                    // a0 -> b0
            BaseAndCount { base: a_arr_p0, count: (a_arr_p1 + 1) - a_arr_p0 },              // a0 -> a1
            BaseAndCount { base: a_arr_p0, count: (b_arr_p1 + 1) - a_arr_p0 },              // a0 -> b1
            BaseAndCount { base: b_arr_p0, count: (b_arr_p1 + 1) - b_arr_p0 },              // b0 -> b1
            BaseAndCount { base: b_arr_p0, count: (a_arr_p1 + 1) - b_arr_p0 },              // b0 -> a1
        ];

        let result_index = (mip_case + arr_case * 4) as usize;
        // TODO(grundlett): [aspect] listed above
        // usize const aspect_mask = ((this->image_aspect & ~slice.image_aspect) != 0);
        let result_rect_n = rect_n[result_index];
        let bc = bc_indices[result_index];
        result.1 = result_rect_n;

        for i in 0..result_rect_n {
            let rect_i = &mut result.0[i];
            let bc_i = bc[i];

            *rect_i = *self;
            // TODO(grundlett): [aspect] listed above
            // rect_i.image_aspect &= ~slice.image_aspect;

            rect_i.base_mip_level = mip_bc[bc_i.0].base;
            rect_i.level_count = mip_bc[bc_i.0].count;
            rect_i.base_array_layer = arr_bc[bc_i.1].base;
            rect_i.layer_count = arr_bc[bc_i.1].base;
        }

        result
    }
}
