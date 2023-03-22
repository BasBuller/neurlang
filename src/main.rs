use neurlang::array::Array;
use neurlang::neurlang::Shape;

fn main() {
    let array = Array::<f32, 2>::new(vec![5.0; 25000000], Shape::new([5000, 5000]));
    for _ in 0..100 {
        let _res = array.permute([1, 0]);
    }
}
