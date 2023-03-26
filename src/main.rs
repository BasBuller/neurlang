#![feature(generic_const_exprs)]

use neurlang::array::Array;
use neurlang::neurlang::{PadAxis, Shape};

fn main() {
    let shape = Shape::new([2, 2, 2]);
    println!("{shape:?}");

    // let values = Array::<f32, 3>::new(vec![1.0], Shape::new([1, 1, 1]));
    // let padding = [
    //     PadAxis::new(1, 1, 0.0),
    //     PadAxis::new(1, 1, 0.0),
    //     PadAxis::new(1, 1, 0.0),
    // ];
    // let padded_values = values.pad(&padding);
    // println!("{:?}", padded_values.values);
    // println!("{}", padded_values.values.borrow().len());
}
