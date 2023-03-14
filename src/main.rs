use neurlang::array::{rand_f32, Array};
use neurlang::indexing::make_slice;
use neurlang::neurlang::Shape;

fn matmul(left_array: &Array<f32>, right_array: &Array<f32>) -> Array<f32> {
    let left_slices =
        (0..left_array.shape.dimensions[1]).map(|index| make_slice(&left_array.shape, 1, index));
    let right_slices =
        (0..right_array.shape.dimensions[0]).map(|index| make_slice(&right_array.shape, 0, index));
    let mut left_right_slices = right_slices.map(|right_slice| {
        left_slices
            .clone()
            .map(move |left_slice| left_slice.into_iter().zip(right_slice.into_iter()))
    });
    let test = left_right_slices
        .next()
        .unwrap()
        .next()
        .unwrap()
        .next()
        .unwrap();
    println!("{:?}", test);

    left_array.clone()

    // let array = self.values.borrow();
    // let slice_iter = make_slice(&self.shape, axis, index).into_iter();
    // let mut res_values = Vec::with_capacity(slice_iter.n_prefix * slice_iter.n_suffix);
    // for (start_idx, end_idx) in slice_iter {
    //     res_values.extend_from_slice(&array[start_idx..end_idx]);
    // }
}

fn main() {
    let l_arr = rand_f32(Shape::new(vec![2, 3]));
    let slice = l_arr.slice(1, 0);
    println!("{:?}", slice.shape);

    let r_arr = rand_f32(Shape::new(vec![3, 4]));
    let slice = r_arr.slice(0, 0);
    println!("{:?}", slice.shape);

    matmul(&l_arr, &r_arr);
}
