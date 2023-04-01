/// Product of a slice of usize, so fold with start value 1
pub fn product(shape: &[usize]) -> usize {
    shape.iter().fold(1, |res, &val| res * val)
}

pub fn permute<T: Copy>(permute_values: &[T], permutation: &[usize]) -> Vec<T> {
    permutation.iter().map(|&idx| permute_values[idx]).collect()
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

pub fn calculate_strides(dimensions: &[usize]) -> Vec<usize> {
    let mut results = vec![1; dimensions.len()];
    for idx in (0..dimensions.len() - 1).rev() {
        results[idx] = results[idx + 1] * dimensions[idx + 1];
    }
    results
}

pub fn linear_to_array_index<const N: usize>(
    linear_index: usize,
    strides: &[usize; N],
) -> [usize; N] {
    let mut results = [1; N];
    let mut index = linear_index;
    for (res_val, &stride) in results.iter_mut().zip(strides.iter()) {
        *res_val = index / stride;
        index %= stride;
    }
    results
}

pub fn array_to_linear_index<const N: usize>(
    array_index: &[usize; N],
    strides: &[usize; N],
) -> usize {
    strides
        .iter()
        .zip(array_index.iter())
        .fold(0, |res, (&lval, &rval)| res + lval * rval)
}

pub fn compare_slices<T: PartialEq>(target: &[T], values: &[T]) {
    let compared = target
        .iter()
        .zip(values.iter())
        .filter(|(targ, val)| targ.eq(val))
        .count();
    assert_eq!(compared, target.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product() {
        assert_eq!(product(&[1, 2, 3, 4]), [1, 2, 3, 4].iter().product());
    }

    #[test]
    fn test_outer_product() {
        let major_vals = vec![vec![1, 2], vec![3, 4]];
        let minor_vals = vec![5, 6];
        let res_values = outer_product(&major_vals, &minor_vals);
        assert_eq!(
            res_values,
            vec![vec![1, 2, 5], vec![1, 2, 6], vec![3, 4, 5], vec![3, 4, 6]]
        );
    }

    #[test]
    fn test_calculate_strides() {
        assert_eq!(calculate_strides(&[1, 2, 3, 4]), [24, 12, 4, 1]);
    }

    #[test]
    fn test_permute() {
        let values = [1, 2, 3, 4];
        let permutation = [1, 0, 3, 2];
        let target = [2, 1, 4, 3];
        let permuted = permute(&values, &permutation);
        compare_slices(&target, &permuted);
    }
}
