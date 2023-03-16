use neurlang::array::*;
use neurlang::indexing::make_slice;
use neurlang::neurlang::Shape;

fn move_dim_forward(array: &Array<f32>, first_dim: usize) -> Vec<(usize, usize)> {
    let nelem = array.shape.borrow().dimensions[first_dim];
    let new_values = (0..nelem)
        .flat_map(|index| make_slice(&array.shape.borrow(), first_dim, index).into_iter())
        .collect::<Vec<_>>();
    new_values
}

fn slice_vector(values: &Vec<f32>, shape: &Shape, axis: usize, index: usize) -> Vec<f32> {
    let slice_iter = make_slice(shape, axis, index).into_iter();
    let mut res_values = Vec::with_capacity(slice_iter.n_prefix * slice_iter.n_suffix);
    for (start_idx, end_idx) in slice_iter {
        res_values.extend_from_slice(&values[start_idx..end_idx]);
    }
    res_values
}

fn permute_naive(values: &Vec<f32>, current_shape: &Shape, new_shape: Vec<usize>) -> Vec<f32> {
    if new_shape.len() == 1 {
        values.clone()
    } else {
        let new_slice_shape = new_shape[1..]
            .iter()
            .map(|&val| if val < new_shape[0] { val } else { val - 1 })
            .collect::<Vec<_>>();
        let mut current_slice_shape = current_shape.dimensions.clone();
        current_slice_shape.remove(new_shape[0]);
        let current_slice_shape = Shape::new(current_slice_shape);

        let dim_size = current_shape.dimensions[new_shape[0]];
        let new_array = (0..dim_size)
            .flat_map(|dim_value| {
                let slice = slice_vector(values, current_shape, new_shape[0], dim_value);
                let slice = permute_naive(&slice, &current_slice_shape, new_slice_shape.clone());
                slice
            })
            .collect::<Vec<_>>();

        new_array
    }
}

fn main() {
    let array = Array::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        Shape::new(vec![2, 2, 2]),
    );
    println!("{:?}", array.values.borrow());

    println!("");

    let indices = move_dim_forward(&array, 2);
    println!("{:?}", indices);
    
    // let values = vec![1.0, 2.0, 3.0, 4.0];
    // let shape = Shape::new(vec![2, 2]);
    // let new_values = permute_naive(&values, &shape, vec![1, 0]);
    // println!("{:?}", values);
    // println!("{:?}", new_values);

    // println!("");

    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape = Shape::new(vec![2, 2, 2]);
    let new_values = permute_naive(&values, &shape, vec![1, 0, 2]);
    println!("{:?}", values);
    println!("{:?}", new_values);
}
