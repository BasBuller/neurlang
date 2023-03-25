use neurlang::array::Array;
use neurlang::neurlang::Shape;

fn main() {
    let values = Array::<f32, 2>::new(vec![1.0; 5000 * 5000], Shape::new([5000, 5000]));
    for _ in 0..10 {
        let new_values = values.permute([1, 0]);
    }
}
