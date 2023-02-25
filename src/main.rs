use neurlang::*;

fn main() {
    let target1: Vec<f32> = vec![-1.0, -2.0, -3.0];
    let arr1 = Array::<f32>::new(vec![1.0, 2.0, 3.0], vec![3]);
    arr1.negate();
    println!("{arr1:?}");
}
