use neurlang::*;

fn main() {
    let array = rand_f32(vec![128, 512, 1024]);
    let _res = array.slice(1, 10);
}
