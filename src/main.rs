use neurlang::*;

fn main() {
    let arr = rand_f32(vec![1, 2]);
    let vec = arr.values.borrow();
    
    
    println!("{arr:?}");
}
