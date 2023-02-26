use neurlang::*;

fn main() {
    let mut arr = vec![1.0, 2.0, 3.0];
    arr.iter_mut().map(|value| *value = -(*value)).count();
    println!("{arr:?}");
}
