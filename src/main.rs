use neurlang::*;
    
// // Reduce
// pub fn reduce(array: &Vec<f32>, shape: &Vec<usize>, axis: usize) -> Vec<f32> {
//     let prefix_numbers = shape[0..axis].iter().fold(1, |res, &value| res * value);
//     let suffix_length = shape[(axis+1)..].iter().fold(1, |res, &value| res * value);
//     let total_stride = prefix_numbers * suffix_length;

//     let mut res_vector = vec![0_f32; total_stride];
//     for reduce_idx in 0..shape[axis] {
//         for prefix_idx in 0..prefix_numbers {
//             total_stride[]
//         }
//     };
//     res_vector
// }

fn slice(array: &Vec<f32>, shape: &Vec<usize>, axis: usize, index: usize) -> Vec<f32> {
    let n_prefix = shape[0..axis].iter().fold(1, |res, &value| res * value);
    let n_axis_suffix = shape[axis..].iter().fold(1, |res, &value| res * value);
    let n_suffix = shape[(axis + 1..)].iter().fold(1, |res, &value| res * value);    
    let mut res = vec![0_f32; n_prefix * n_suffix];

    for prefix_idx in 0..n_prefix {
        for suffix_idx in 0..n_suffix{
            let res_idx = (prefix_idx * n_suffix) + suffix_idx;
            let arr_idx = (prefix_idx * n_axis_suffix) + (index * n_suffix) + suffix_idx;
            res[res_idx] = array[arr_idx];
        }
    }
    res
}

fn main() {
    let shape = vec![2, 2, 2];
    let n_elems = shape.iter().fold(1, |res, value| res * value);
    let vec0 = (1..(n_elems + 1)).map(|val| val as f32).collect::<Vec<f32>>();
    println!("{vec0:?}\n");
    
    let idx: usize = 2;
    println!("Index {idx}, slice 0: {:?}\n", slice(&vec0, &shape, idx, 0));
    println!("Index {idx}, slice 1: {:?}\n", slice(&vec0, &shape, idx, 1));

    // for idx in 0..shape.len() {
    //     println!("\nIndex {idx}, slice 0: {:?}", slice(&vec0, &shape, idx, 0));
    //     println!("Index {idx}, slice 1: {:?}", slice(&vec0, &shape, idx, 1));
    // }
}
