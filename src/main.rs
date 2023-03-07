use neurlang::neurlang::*;
use neurlang::slice::*;

fn main() {
    for idx in 0..4 {
        let shape = vec![2, 2, 2, 2];
        let slice = make_slice(&shape, idx, 1);
        let iter = slice.into_iter().collect::<Vec<_>>();
        println!("{iter:?}");
    }
}
