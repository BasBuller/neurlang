pub fn count_elements(shape: &[usize]) -> usize {
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
