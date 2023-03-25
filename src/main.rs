#![feature(generic_const_exprs)]

use neurlang::array::Array;
use neurlang::neurlang::{PadAxis, Shape};

fn main() {
    let values = Array::<f32, 2>::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let padding = [PadAxis::new(1, 1, 0.0), PadAxis::new(1, 1, 0.0)];
    let padded_values = values.pad(&padding);
    println!("{:?}", padded_values.values);
}
