use neurlang::*;

fn main() {
    let slice = make_slice(vec![2, 2, 2], 0, 0);
    let iter = slice.into_iter().collect::<Vec<_>>();
    println!("{iter:?}");
}
