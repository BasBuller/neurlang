use neurlang::array::*;
use neurlang::neurlang::*;

fn main() {
    let side_len: usize = 3500;
    let shape0 = Shape::new(vec![side_len, side_len]);
    let arr0: Array<f32> = Array::new(vec![0.0; side_len * side_len]);
    let shape1 = Shape::new(vec![side_len, side_len]);
    let arr1: Array<f32> = Array::new(vec![0.0; side_len * side_len]);
    let res = arr0.matmul(&shape0, &arr1, &shape1);
    let slice = &res.values.borrow()[0..5];
    println!("{:?}", slice);
}
