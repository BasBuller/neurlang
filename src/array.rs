use crate::indexing::*;
use crate::neurlang::{ExecuteAST, PadAxis, Padding, ReduceAxis, Shape};
use crate::utils::revert_permute;

use num::Float;
use rand::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Array<T>
where
    T: Float,
{
    pub values: Rc<RefCell<Vec<T>>>,
}

pub fn rand_f32(shape: &Shape) -> Array<f32> {
    let mut rng = rand::thread_rng();
    let total_elems = shape.nelem();
    let values = (0..total_elems).map(|_| rng.gen()).collect::<Vec<_>>();
    Array::new(values)
}

impl<T> Array<T>
where
    T: Float + std::fmt::Debug + Default + std::iter::Sum,
{
    // Utils
    pub fn new(values: Vec<T>) -> Self {
        Array {
            values: Rc::new(RefCell::new(values)),
        }
    }
    pub fn reference(values: Rc<RefCell<Vec<T>>>) -> Self {
        Array { values: values }
    }

    // Unary
    pub fn unary_op<F>(&self, unary_f: F) -> Self
    where
        F: Fn(&T) -> T,
    {
        let iterated = self.values.borrow().iter().map(unary_f).collect::<Vec<_>>();
        Self::new(iterated)
    }
    pub fn negate(&self) -> Self {
        self.unary_op(|&value| -value)
    }
    pub fn exp(&self) -> Self {
        self.unary_op(|value| value.exp())
    }
    pub fn ln(&self) -> Self {
        self.unary_op(|value| value.ln())
    }

    pub fn inpl_unary_op<F>(&self, unary_f: F)
    where
        F: Fn(&mut T),
    {
        self.values.borrow_mut().iter_mut().for_each(unary_f);
    }
    pub fn inpl_negate(&self) {
        self.inpl_unary_op(|value| *value = -(*value));
    }
    pub fn inpl_exp(&self) {
        self.inpl_unary_op(|value| *value = value.exp());
    }
    pub fn inpl_ln(&self) {
        self.inpl_unary_op(|value| *value = value.ln());
    }

    // Binary
    pub fn binary_op<F>(&self, right_array: &Self, binary_f: F) -> Self
    where
        F: Fn((&T, &T)) -> T,
    {
        let res_values = self
            .values
            .borrow()
            .iter()
            .zip(right_array.values.borrow().iter())
            .map(binary_f)
            .collect::<Vec<_>>();
        Self::new(res_values)
    }
    pub fn add(&self, right_array: &Self) -> Self {
        self.binary_op(right_array, |(&lval, &rval)| lval + rval)
    }
    pub fn sub(&self, right_array: &Self) -> Self {
        self.binary_op(right_array, |(&lval, &rval)| lval - rval)
    }
    pub fn multiply(&self, right_array: &Self) -> Self {
        self.binary_op(right_array, |(&lval, &rval)| lval * rval)
    }
    pub fn divide(&self, right_array: &Self) -> Self {
        self.binary_op(right_array, |(&lval, &rval)| lval / rval)
    }
    pub fn pow(&self, right_array: &Self) -> Self {
        self.binary_op(right_array, |(&lval, &rval)| lval.powf(rval))
    }
    pub fn compare_equal(&self, right_array: &Self) -> Self {
        self.binary_op(
            right_array,
            |(&lval, &rval)| if lval == rval { T::one() } else { T::zero() },
        )
    }
    pub fn max(&self, right_array: &Self) -> Self {
        self.binary_op(
            right_array,
            |(&lval, &rval)| if lval > rval { lval } else { rval },
        )
    }

    // Axis reducing operations
    fn slice_vector(&self, shape: &Shape, axis: usize, index: usize) -> Vec<T> {
        let values = self.values.borrow();
        let n_prefix_items: usize = shape.dimensions[0..axis].iter().product();
        let axis_stride: usize  = shape.dimensions[axis..].iter().product();
        let suffix_stride: usize = shape.dimensions[(axis + 1)..].iter().product();

        let mut slice_values = Vec::with_capacity(n_prefix_items * suffix_stride);
        for prefix_idx in 0..n_prefix_items {
            let src_start_idx = prefix_idx * axis_stride + index * suffix_stride;
            let src_end_idx = src_start_idx + suffix_stride;
            slice_values.extend_from_slice(&values[src_start_idx..src_end_idx]);
        }

        slice_values
    }

    pub fn slice(&self, shape: &Shape, axis: usize, index: usize) -> Array<T> {
        let res_values = self.slice_vector(shape, axis, index);
        Array::new(res_values)
    }

    pub fn reduce<F>(&self, shape: &Shape, axis: usize, reduce_f: F) -> Array<T>
    where
        F: Fn((&mut T, &T)),
    {
        let n_prefix = shape.dimensions[0..axis].iter().product();
        let n_axis_suffix = shape.dimensions[axis..].iter().product::<usize>();
        let n_suffix = shape.dimensions[axis + 1..].iter().product::<usize>();
        let array = self.values.borrow();

        let mut res_values = self.slice_vector(shape, axis, 0);
        for prefix_idx in 0..n_prefix {
            for index in 1..shape.dimensions[axis] {
                let src_start_idx = (prefix_idx * n_axis_suffix) + (index * n_suffix);
                let src_end_idx = src_start_idx + n_suffix;
                let src_slice = &array[src_start_idx..src_end_idx];

                let res_start_idx = prefix_idx * n_suffix;
                let res_end_idx = res_start_idx + n_suffix;
                let res_slice = &mut res_values[res_start_idx..res_end_idx];

                res_slice
                    .iter_mut()
                    .zip(src_slice.iter())
                    .map(&reduce_f)
                    .count();
            }
        }
        Array::new(res_values)
    }

    pub fn reduce_sum(&self, shape: &Shape, axis: usize) -> Array<T> {
        self.reduce(shape, axis, |(res_val, src_val)| {
            *res_val = *res_val + *src_val
        })
    }
    pub fn reduce_max(&self, shape: &Shape, axis: usize) -> Array<T> {
        self.reduce(shape, axis, |(res_val, src_val)| {
            *res_val = if *res_val > *src_val {
                *res_val
            } else {
                *src_val
            }
        })
    }

    // Movement ops
    pub fn unsqueeze(&self, shape: &Shape, axis: usize) -> Array<T> {
        Array::reference(self.values.clone())
    }
    pub fn squeeze(&self, shape: &Shape, axis: usize) -> Array<T> {
        Array::reference(self.values.clone())
    }

    pub fn reshape(&self, shape: &Shape, new_shape: &[usize]) -> Array<T> {
        Array::reference(self.values.clone())
    }

    pub fn permute(&self, shape: &Shape, permutation: &[usize]) -> Self {
        let cur_values = self.values.borrow();
        let permuted_shape = shape.permute(permutation);
        let reverted_permuted_strides = revert_permute(&permuted_shape.strides, permutation);

        let mut results = Vec::with_capacity(cur_values.len());
        unsafe {
            results.set_len(cur_values.len());
        }
        for (idx, &value) in cur_values.iter().enumerate() {
            let permuted_linear_index =
                shape.linear_permute_linear(idx, &reverted_permuted_strides);
            results[permuted_linear_index] = value;
        }

        Self::new(results)
    }

    pub fn pad(&self, shape: &Shape, axes_padding: Vec<PadAxis<T>>) -> Self {
        let padding_helper = Padding::new(axes_padding, shape);
        let new_nelem = padding_helper.padded_sizes.iter().product();
        let mut new_values = Vec::with_capacity(new_nelem);
        padding_helper.pad_array(&mut new_values, &self.values.borrow(), 0);
        Array::new(new_values)
    }

    pub fn strided(&self, shape: &Shape, strides: &[usize]) -> Self {
        let strided_shape = shape.strided(strides);
        let new_shape = Shape::new(strided_shape.dimensions.clone());
        let new_n_elem = strided_shape.nelem();
        let values = self.values.borrow();

        let mut new_values = Vec::with_capacity(new_n_elem);
        for linear_idx in 0..new_n_elem {
            let array_index = new_shape.linear_to_array_index(linear_idx);
            let strided_index = strided_shape.array_to_linear_index(&array_index);
            new_values.extend_from_slice(&values[strided_index..(strided_index + 1)]);
        }

        Array::new(new_values)
    }

    pub fn matmul(&self, left_shape: &Shape, right_array: &Array<T>, right_shape: &Shape) -> Self {
        let left_n_iters: usize = left_shape.dimensions[0..(left_shape.len() - 1)].iter().product();
        let left_chunk_size = left_shape.dimensions[left_shape.dimensions.len() - 1];
        let right_n_iters: usize = right_shape.dimensions[1..].iter().product();
        let right_shape_flat = Shape::new(vec![right_shape.dimensions[0], right_n_iters]);
        
        let left_values = self.values.borrow();
        let mut res_values = Vec::with_capacity(left_n_iters * right_n_iters);
        for right_iter in 0..right_n_iters {
            let right_vector = right_array.slice_vector(&right_shape_flat, 1, right_iter);
            for left_vector in left_values.chunks(left_chunk_size) {
                let value: T = right_vector.iter().zip(left_vector.iter()).map(|(&l, &r)| l * r).sum();
                res_values.push(value);
            }
        }

        Array::new(res_values)
    }
}

impl<T> ExecuteAST<T> for Array<T>
where
    T: Float + std::fmt::Debug + Default + std::iter::Sum,
{
    // Leaf
    fn value_v(&self) -> Self {
        self.clone()
    }

    // Unary
    fn negate_v(&self) -> Self {
        self.negate()
    }
    fn exp_v(&self) -> Self {
        self.exp()
    }
    fn log_v(&self) -> Self {
        self.ln()
    }

    // Binary
    fn add_v(&self, right_value: &Self) -> Self {
        self.add(right_value)
    }
    fn sub_v(&self, right_value: &Self) -> Self {
        self.sub(right_value)
    }
    fn mul_v(&self, right_value: &Self) -> Self {
        self.multiply(right_value)
    }
    fn div_v(&self, right_value: &Self) -> Self {
        self.divide(right_value)
    }
    fn pow_v(&self, right_value: &Self) -> Self {
        self.pow(right_value)
    }
    fn compare_equal_v(&self, right_value: &Self) -> Self {
        self.compare_equal(right_value)
    }
    fn max_v(&self, right_value: &Self) -> Self {
        self.max(right_value)
    }

    // Reduce
    fn reduce_max_v(&self, shape: &Shape, axis: ReduceAxis) -> Self {
        self.reduce_max(shape, axis)
    }
    fn reduce_sum_v(&self, shape: &Shape, axis: ReduceAxis) -> Self {
        self.reduce_sum(shape, axis)
    }

    // Movement
    fn unsqueeze_v(&self, shape: &Shape, dim: usize) -> Self {
        self.unsqueeze(shape, dim)
    }
    fn squeeze_v(&self, shape: &Shape, dim: usize) -> Self {
        self.squeeze(shape, dim)
    }
    fn reshape_v(&self, shape: &Shape, new_shape: &[usize]) -> Self {
        self.reshape(shape, new_shape)
    }
    fn permute_v(&self, shape: &Shape, axis_ordering: &[usize]) -> Self {
        self.permute(shape, axis_ordering)
    }
    fn pad_v(&self, shape: &Shape, axes_padding: Vec<PadAxis<T>>) -> Self {
        self.pad(shape, axes_padding)
    }

    // Higher order
    fn matmul_v(&self, left_shape: &Shape, right_value: &Self, right_shape: &Shape) -> Self {
        self.matmul(left_shape, right_value, right_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{calculate_strides, compare_slices};

    #[test]
    fn test_rolling_dimensions_lengths() {
        let shape = [2, 2, 2];
        let lengths = calculate_strides(&shape);
        let target = [4, 2, 1];
        compare_slices(&target, &lengths);
    }

    #[test]
    fn negate() {
        let target: Vec<f32> = vec![-1.0, -2.0, -3.0];
        let arr1 = Array::<f32>::new(vec![1.0, 2.0, 3.0]).negate();
        compare_slices(&target, &arr1.values.borrow());
    }

    #[test]
    fn add() {
        let target: Vec<f32> = vec![5.0, 7.0, 9.0];
        let arr1 = Array::<f32>::new(vec![1.0, 2.0, 3.0]);
        let arr2 = Array::<f32>::new(vec![4.0, 5.0, 6.0]);
        let arr3 = arr1.add(&arr2);
        compare_slices(&target, &arr3.values.borrow());
    }

    #[test]
    fn slice() {
        let shape = Shape::new(vec![2, 2, 2]);
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let arr0 = arr.slice_vector(&shape, 0, 0);
        let target0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        compare_slices(&target0, &arr0);

        let arr1 = arr.slice_vector(&shape, 1, 0);
        let target1: Vec<f32> = vec![1.0, 2.0, 5.0, 6.0];
        compare_slices(&target1, &arr1);

        let arr2 = arr.slice_vector(&shape, 2, 0);
        let target2: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0];
        compare_slices(&target2, &arr2);
    }

    #[test]
    fn reduce_sum() {
        let shape = Shape::new(vec![2, 2, 2]);
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let arr0 = arr.reduce_sum(&shape, 0);
        let target0: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];
        compare_slices(&target0, &arr0.values.borrow());

        let arr1 = arr.reduce_sum(&shape, 1);
        let target1: Vec<f32> = vec![4.0, 6.0, 12.0, 14.0];
        compare_slices(&target1, &arr1.values.borrow());

        let arr2 = arr.reduce_sum(&shape, 2);
        let target2: Vec<f32> = vec![3.0, 7.0, 11.0, 15.0];
        compare_slices(&target2, &arr2.values.borrow());
    }

    #[test]
    fn reduce_max() {
        let shape = Shape::new(vec![2, 2, 2]);
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let arr0 = arr.reduce_max(&shape, 0);
        let target0: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        compare_slices(&target0, &arr0.values.borrow());

        let arr1 = arr.reduce_max(&shape, 1);
        let target1: Vec<f32> = vec![3.0, 4.0, 7.0, 8.0];
        compare_slices(&target1, &arr1.values.borrow());

        let arr2 = arr.reduce_max(&shape, 2);
        let target2: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
        compare_slices(&target2, &arr2.values.borrow());
    }

    #[test]
    fn matmul() {
        let l_shape = Shape::new(vec![2, 3]);
        let l_arr = Array::<f32>::new(vec![2.0; 6]);
        let r_shape = Shape::new(vec![3, 4]);
        let r_arr = Array::<f32>::new(vec![3.0; 12]);
        let res = l_arr.matmul(&l_shape, &r_arr, &r_shape);
        let target_values = vec![18.0; 8];
        compare_slices(&target_values, &res.values.borrow());
    }

    #[test]
    fn permute() {
        let shape = Shape::new(vec![2, 2]);
        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0]);
        let new_values = values.permute(&shape, &[1, 0]);
        let target = vec![1.0, 3.0, 2.0, 4.0];
        compare_slices(&target, &new_values.values.borrow());

        let shape = Shape::new(vec![2, 2, 2]);
        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let new_values = values.permute(&shape, &[1, 0, 2]);
        let target = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
        compare_slices(&target, &new_values.values.borrow());

        let shape = Shape::new(vec![2, 2, 2]);
        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let new_values = values.permute(&shape, &[2, 1, 0]);
        let target = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
        compare_slices(&target, &new_values.values.borrow());

        let shape = Shape::new(vec![2, 2]);
        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0]);
        let new_values = values.permute(&shape, &[1, 0]);
        let target = vec![1.0, 3.0, 2.0, 4.0];
        compare_slices(&target, &new_values.values.borrow());

        let shape = Shape::new(vec![1, 2, 3]);
        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let new_values = values.permute(&shape, &[1, 2, 0]);
        let target = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        compare_slices(&target, &new_values.values.borrow());
    }

    #[test]
    fn padding() {
        let shape = Shape::new(vec![2, 2]);
        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0]);
        let padding = vec![PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let padded_values = values.pad(&shape, padding);
        let target = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        compare_slices(&padded_values.values.borrow(), &target);

        let shape = Shape::new(vec![1, 1, 1]);
        let values = Array::<f32>::new(vec![1.0]);
        let padding = vec![PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let padded_values = values.pad(&shape, padding);
        let target = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        compare_slices(&padded_values.values.borrow(), &target);
    }
    
    #[test]
    fn strides() {
        let shape = Shape::new(vec![2, 4]);
        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let strides = [1, 2];
        let strides_values = values.strided(&shape, &strides);
        let target_values = vec![1.0, 3.0, 5.0, 7.0];
        compare_slices(&target_values, &strides_values.values.borrow());

        let strides = [2, 1];
        let strides_values = values.strided(&shape, &strides);
        let target_values = vec![1.0, 2.0, 3.0, 4.0];
        compare_slices(&target_values, &strides_values.values.borrow());
    }
}
