use neurlang::neurlang::Shape;
use neurlang::array::Array;

fn main() {
    let values = Array::<f32, 2>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3]));
    let new_values = values.permute([1, 0]);
    println!("{:?}", new_values.values.borrow());
}
