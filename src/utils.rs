/// Product of a slice of usize, so fold with start value 1
pub fn product(shape: &[usize]) -> usize {
    shape.iter().fold(1, |res, &val| res * val)
}

pub fn outer_product<T: Copy + Sized>(major_vals: &Vec<Vec<T>>, minor_vals: &[T]) -> Vec<Vec<T>> {
    let mut res_values: Vec<Vec<T>> = Vec::with_capacity(major_vals.len() * minor_vals.len());
    for major_val in major_vals.iter() {
        for &minor_val in minor_vals.iter() {
            let mut prod: Vec<T> = major_val.clone().into();
            prod.push(minor_val);
            res_values.push(prod);
        }
    }
    res_values
}

pub fn rolling_dimensions_lengths<const N: usize>(dimensions: &[usize; N]) -> [usize; N] {
    let mut results = [1; N];
    for idx in 1..N {
        results[idx - 1] = dimensions[idx..].iter().fold(1, |res, &val| res * val);
    }
    results
}
