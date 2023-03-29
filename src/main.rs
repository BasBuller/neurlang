#![feature(generic_const_exprs)]

use neurlang::array::Array;
use neurlang::neurlang::Shape;

fn main() {
    let array = Array::new(vec![1.0; 5000 * 5000], Shape::new([5000, 5000]));
    for _ in 0..1 {
        let _ = array.permute([1, 0]);
    }
}
