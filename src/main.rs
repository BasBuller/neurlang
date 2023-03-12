use neurlang::outer_product;

fn main() {
    let indices = outer_product(&vec![vec![0], vec![1]], &vec![0, 1]);
    let indices = outer_product(&indices, &vec![0, 1]);
    println!("{indices:?}");
}
