use neurlang::*;

fn main() {
    let shape = vec![1, 2, 3];
    let n_elems = shape.iter().fold(1, |res, value| res * value);
    let vec0 = (1..(n_elems + 1))
        .map(|val| val as f32)
        .collect::<Vec<f32>>();
    let arr0 = Array::new(vec0, shape);
    println!("{arr0:?}");

    let arr1 = arr0.sum(1);
    println!("{arr1:?}");
}
