use neurlang::indexing::PermutedTensorIterator;
use neurlang::neurlang::Shape;

fn main() {
    let shape = Shape::new(vec![2, 2, 2]);
    let permutation = [1, 0, 2];
    let iterator = PermutedTensorIterator::new(shape, permutation);
    let results = iterator.into_iter().collect::<Vec<_>>();
    println!("{:?}", results);
}
