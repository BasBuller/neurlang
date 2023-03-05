use neurlang::*;

fn main() {
    let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let summed_arr = arr.sum(2);
    println!("\n{:?}", summed_arr);
}
