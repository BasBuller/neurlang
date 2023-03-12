use neurlang::indexing::TensorIterator;
use neurlang::neurlang::{MemoryLayout, Shape};

fn main() {
    let shape = Shape::new(vec![2, 2, 2, 2]);
    let layout = MemoryLayout::RowMajor;
    let tensor_iter: TensorIterator<4> = TensorIterator::new(shape, layout);
    let indices = tensor_iter.into_iter().collect::<Vec<_>>();
    println!("{indices:?}");
}
