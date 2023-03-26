#![feature(generic_const_exprs)]

use neurlang::array::Array;
use neurlang::neurlang::{PadAxis, Shape};
use num::Float;

// fn permute_tensor<T: Float + Default>(tensor: &[T], shape: &[usize], permutation: &[usize]) -> Vec<T>
// where
//     T: Clone,
// {
//     assert_eq!(shape.len(), permutation.len(), "Shape and permutation lengths do not match");

//     // Calculate the strides of the original tensor
//     let mut strides = vec![1; shape.len()];
//     for i in (0..shape.len() - 1).rev() {
//         strides[i] = strides[i + 1] * shape[i + 1];
//     }

//     // Calculate the new shape and strides of the permuted tensor
//     let mut new_shape = vec![0; shape.len()];
//     let mut new_strides = vec![0; shape.len()];
//     for (i, &permuted_axis) in permutation.iter().enumerate() {
//         new_shape[i] = shape[permuted_axis];
//         new_strides[i] = strides[permuted_axis];
//     }

//     let mut permuted_tensor = vec![Default::default(); tensor.len()];
//     for (new_idx, new_offset) in compute_offsets(&new_shape, &new_strides).enumerate() {
//         let mut old_offset = 0;
//         for (i, &permuted_axis) in permutation.iter().enumerate() {
//             let idx = new_idx / new_strides[i] % new_shape[i];
//             old_offset += idx * strides[permuted_axis];
//         }
//         permuted_tensor[new_offset] = tensor[old_offset].clone();
//     }

//     permuted_tensor
// }

// // Helper function to compute offsets given a shape and strides
// fn compute_offsets(shape: &[usize], strides: &[usize]) -> impl Iterator<Item = usize> {
//     (0..shape.iter().product()).map(move |idx| {
//         let mut offset: usize = 0;
//         for (i, &stride) in strides.iter().enumerate() {
//             offset += (idx / shape[i..].iter().product()) % shape[i] * stride;
//         }
//         offset
//     })
// }


fn main() {
    // let values = vec![1.0; 8];
    // let shape = vec![2, 2, 2];
    // let permutation = vec![1, 2, 0];
    // let permuted_tensor = permute_tensor(&values, &shape, &permutation);
}
