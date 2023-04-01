use crate::indexing::*;
use crate::neurlang::{ExecuteAST, MemoryLayout, ReduceAxis, ReduceOp};
use crate::utils::{calculate_strides, permute, permute_with_target};

use num::Float;
use rand::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct NewAxis {
    index: usize,
    axis_size: usize,
}
impl NewAxis {
    pub fn new(index: usize, axis_size: usize) -> Self {
        NewAxis {
            index: index,
            axis_size: axis_size,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ArrayIndex {
    pub index: Vec<usize>,
}
impl ArrayIndex {
    pub fn new(index: Vec<usize>) -> Self {
        ArrayIndex { index }
    }
}

/// Contains:
///     1. Prefix padding count
///     2. Suffix padding count
///     3. Padding value)
#[derive(Debug, Clone, Copy)]
pub struct PadAxis<T>(pub usize, pub usize, pub T);
pub struct Padding<T> {
    pub axes_padding: Vec<PadAxis<T>>,
    pub padded_sizes: Vec<usize>,
    pub padded_strides: Vec<usize>,
    pub original_strides: Vec<usize>,
}
impl<T: Clone + Copy> Padding<T> {
    pub fn new(axes_padding: Vec<PadAxis<T>>, shape: &Shape) -> Self {
        let mut padded_sizes = shape.dimensions.clone();
        padded_sizes.iter_mut().zip(axes_padding.iter()).for_each(
            |(size, PadAxis(prefix_count, suffix_count, _))| {
                *size = *size + prefix_count + suffix_count
            },
        );
        let mut padded_strides = vec![1; shape.len()];
        for idx in 1..shape.len() {
            padded_strides[idx - 1] = padded_sizes[idx..].iter().fold(1, |res, &val| res * val);
        }
        Padding {
            axes_padding,
            padded_sizes,
            padded_strides,
            original_strides: shape.strides.clone(),
        }
    }

    // TODO: Optimize this function further by collapsing dimensions that are not padded into the previous dimension, this enables larger chunks being transferred at once
    pub fn pad_array(&self, new_values: &mut Vec<T>, original_values: &[T], axis_index: usize) {
        let padded_stride = self.padded_strides[axis_index];
        let original_stride = self.original_strides[axis_index];
        let n_prefix = padded_stride * self.axes_padding[axis_index].0;
        let n_suffix = padded_stride * self.axes_padding[axis_index].1;
        let pad_val = self.axes_padding[axis_index].2;

        new_values.resize(new_values.len() + n_prefix, pad_val);
        if axis_index < self.axes_padding.len() - 1 {
            for original_values_chunk in original_values.chunks(original_stride) {
                self.pad_array(new_values, original_values_chunk, axis_index + 1);
            }
        } else {
            new_values.extend_from_slice(&original_values);
        };
        new_values.resize(new_values.len() + n_suffix, pad_val);
    }
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub dimensions: Vec<usize>,
    pub strides: Vec<usize>,
}
impl Shape {
    pub fn new(dimensions: Vec<usize>) -> Self {
        let strides = calculate_strides(&dimensions);
        Shape {
            dimensions,
            strides,
        }
    }

    pub fn remove(&self, index: ReduceAxis) -> Shape {
        let mut new_dimensions = Vec::with_capacity(self.dimensions.len() - 1);
        new_dimensions.extend_from_slice(&self.dimensions[0..index]);
        new_dimensions.extend_from_slice(&self.dimensions[(index + 1)..]);
        Shape::new(new_dimensions)
    }

    pub fn insert(&self, new_axis: NewAxis) -> Shape {
        let mut new_dimensions = Vec::with_capacity(self.dimensions.len() + 1);
        new_dimensions.extend_from_slice(&self.dimensions[0..new_axis.index]);
        new_dimensions.push(new_axis.axis_size);
        new_dimensions.extend_from_slice(&self.dimensions[new_axis.index..]);
        Shape::new(new_dimensions)
    }

    pub fn permute(&self, new_order: &[usize]) -> Self {
        let new_dimensions = permute(&self.dimensions, new_order);
        Self::new(new_dimensions)
    }

    pub fn len(&self) -> usize {
        self.dimensions.len()
    }

    pub fn nelem(&self) -> usize {
        self.dimensions.iter().product()
    }

    pub fn array_to_linear_index(&self, array_index: &[usize]) -> usize {
        self.strides
            .iter()
            .zip(array_index.iter())
            .fold(0, |res, (&lval, &rval)| res + lval * rval)
    }
    
    pub fn linear_to_array_index_with_target(&self, linear_index: usize, target_slice: &mut [usize]) {
        let mut counter = linear_index;
        for (target, &size) in target_slice.iter_mut().zip(self.strides.iter()) {
            *target = counter / size;
            counter = counter % size;
        }
    }

    pub fn linear_to_array_index(&self, linear_index: usize) -> Vec<usize> {
        let mut results = Vec::with_capacity(self.dimensions.len());
        let mut counter = linear_index;
        for &size in self.strides.iter() {
            results.push(counter / size);
            counter = counter % size;
        }
        results
    }
}

#[derive(Debug, Clone)]
pub struct Array<T>
where
    T: Float,
{
    pub values: Rc<RefCell<Vec<T>>>,
    pub shape: RefCell<Shape>,
    pub layout: MemoryLayout,
}

pub fn rand_f32(shape: Shape) -> Array<f32> {
    let mut rng = rand::thread_rng();
    let total_elems = shape.nelem();
    let values = (0..total_elems).map(|_| rng.gen()).collect::<Vec<_>>();
    Array::new(values, shape)
}

impl<T> Array<T>
where
    T: Float + std::fmt::Debug + Default,
{
    // Utils
    pub fn new(values: Vec<T>, shape: Shape) -> Self {
        Array {
            values: Rc::new(RefCell::new(values)),
            shape: RefCell::new(shape),
            layout: MemoryLayout::RowMajor,
        }
    }
    pub fn reference_values(values: Rc<RefCell<Vec<T>>>, shape: Shape) -> Self {
        Array {
            values: values,
            shape: RefCell::new(shape),
            layout: MemoryLayout::RowMajor,
        }
    }
    fn duplicate(&self, values: Vec<T>) -> Self {
        Array {
            values: Rc::new(RefCell::new(values)),
            shape: self.shape.clone(),
            layout: self.layout.clone(),
        }
    }

    // Unary
    pub fn unary_op<F>(&self, unary_f: F) -> Self
    where
        F: Fn(&T) -> T,
    {
        let iterated = self.values.borrow().iter().map(unary_f).collect::<Vec<_>>();
        self.duplicate(iterated)
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
        self.values.borrow_mut().iter_mut().map(unary_f).count();
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
        self.duplicate(res_values)
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
    fn slice_vector(&self, axis: usize, index: usize) -> Vec<T> {
        let array = self.values.borrow();
        let slice_iter = make_slice(&self.shape.borrow(), axis, index).into_iter();
        let mut res_values = Vec::with_capacity(slice_iter.n_prefix * slice_iter.n_suffix);
        for (start_idx, end_idx) in slice_iter {
            res_values.extend_from_slice(&array[start_idx..end_idx]);
        }

        res_values
    }

    pub fn slice(&self, axis: usize, index: usize) -> Array<T> {
        let res_shape = self.shape.borrow().remove(axis);
        let res_values = self.slice_vector(axis, index);
        Array::new(res_values, res_shape)
    }

    pub fn reduce<F>(&self, axis: usize, reduce_f: F) -> Array<T>
    where
        F: Fn((&mut T, &T)),
    {
        let n_prefix = self.shape.borrow().dimensions[0..axis].iter().product();
        let n_axis_suffix = self.shape.borrow().dimensions[axis..]
            .iter()
            .product::<usize>();
        let n_suffix = self.shape.borrow().dimensions[axis + 1..]
            .iter()
            .product::<usize>();
        let array = self.values.borrow();

        let res_shape = self.shape.borrow().remove(axis);
        let mut res_values = self.slice_vector(axis, 0);
        for prefix_idx in 0..n_prefix {
            for index in 1..self.shape.borrow().dimensions[axis] {
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
        Array::new(res_values, res_shape)
    }

    pub fn reduce_sum(&self, axis: usize) -> Array<T> {
        self.reduce(axis, |(res_val, src_val)| *res_val = *res_val + *src_val)
    }
    pub fn reduce_max(&self, axis: usize) -> Array<T> {
        self.reduce(axis, |(res_val, src_val)| {
            *res_val = if *res_val > *src_val {
                *res_val
            } else {
                *src_val
            }
        })
    }

    // Movement ops
    pub fn unsqueeze(&self, axis: usize) -> Array<T> {
        let new_shape = self.shape.borrow().insert(NewAxis::new(axis, 1));
        Array::reference_values(self.values.clone(), new_shape)
    }
    pub fn squeeze(&self, axis: usize) -> Array<T> {
        let new_shape = self.shape.borrow().remove(axis);
        Array::reference_values(self.values.clone(), new_shape)
    }

    pub fn reshape(&self, new_shape: Shape) -> Array<T> {
        Array::<T>::reference_values(self.values.clone(), new_shape)
    }

    pub fn permute(&self, permutation: &[usize]) -> Self {
        let cur_values = self.values.borrow();
        let shape = self.shape.borrow();
        let permuted_shape = shape.permute(&permutation);

        let mut temp_ordered_index = vec![0; shape.len()];
        let mut temp_permuted_index = vec![0; shape.len()];
        let mut results = vec![Default::default(); cur_values.len()];
        for (idx, &value) in cur_values.iter().enumerate() {
            shape.linear_to_array_index_with_target(idx, &mut temp_ordered_index);
            permute_with_target(&temp_ordered_index, &mut temp_permuted_index, &permutation);
            let permuted_linear_index = permuted_shape.array_to_linear_index(&temp_permuted_index);
            results[permuted_linear_index] = value;
        }

        Self::new(results, permuted_shape)
    }

    pub fn pad(&self, axes_padding: Vec<PadAxis<T>>) -> Self {
        let padding_helper = Padding::new(axes_padding, &self.shape.borrow());
        let new_nelem = padding_helper.padded_sizes.iter().product();
        let mut new_values = Vec::with_capacity(new_nelem);
        padding_helper.pad_array(&mut new_values, &self.values.borrow(), 0);
        Array::new(new_values, Shape::new(padding_helper.padded_sizes))
    }

    // pub fn stride(&self, strides: [usize; N]) -> Self {
    //     let mut new_dimensions = self.shape.borrow().dimensions.clone();
    //     for (n_dim, &stride_mul) in new_dimensions.iter_mut().zip(strides.iter()) {
    //         *n_dim /= stride_mul;
    //     }

    //     let mut long_strides = self.shape.borrow().strides.clone();
    //     for (n_stride, &stride_mul) in long_strides.iter_mut().zip(strides.iter()) {
    //         *n_stride *= stride_mul;
    //     }
    //     let new_n_elem = new_dimensions.iter().product();
    //     let mut new_values = Vec::with_capacity(new_n_elem);
    //     for linear_idx in 0..new_n_elem {
    //         let array_index
    //     }
    //     // let new_values =

    //     Array::new(self.values.borrow().clone(), Shape::new(new_dimensions))
    // }

    // Higher order ops
    pub fn matmul(&self, right_array: &Array<T>) -> Self {
        self.clone()
    }
}

impl<T> ExecuteAST<T> for Array<T>
where
    T: Float + std::fmt::Debug + Default,
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
    fn reduce_max_v(&self, axis: ReduceAxis) -> Self {
        self.reduce_max(axis)
    }
    fn reduce_sum_v(&self, axis: ReduceAxis) -> Self {
        self.reduce_sum(axis)
    }
    
    // Movement
    fn unsqueeze_v(&self, dim: usize) -> Self {
        self.unsqueeze(dim)
    }
    fn squeeze_v(&self, dim: usize) -> Self {
        self.squeeze(dim)
    }
    fn reshape_v(&self, new_shape: Shape) -> Self {
        self.reshape(new_shape)
    }
    fn permute_v(&self, axis_ordering: &[usize]) -> Self {
        self.permute(axis_ordering)
    }
    fn pad_v(&self, axes_padding: &Vec<PadAxis<T>>) -> Self {
        self.pad(axes_padding.clone())
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
        let shape = Shape::new(vec![3]);
        let arr1 = Array::<f32>::new(vec![1.0, 2.0, 3.0], shape).negate();
        compare_slices(&target, &arr1.values.borrow());
    }

    #[test]
    fn add() {
        let target: Vec<f32> = vec![5.0, 7.0, 9.0];
        let shape = Shape::new(vec![3]);
        let arr1 = Array::<f32>::new(vec![1.0, 2.0, 3.0], shape.clone());
        let arr2 = Array::<f32>::new(vec![4.0, 5.0, 6.0], shape);
        let arr3 = arr1.add(&arr2);
        compare_slices(&target, &arr3.values.borrow());
    }

    #[test]
    fn slice() {
        let shape = Shape::new(vec![2, 2, 2]);
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape);

        let arr0 = arr.slice(0, 0);
        let target0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        compare_slices(&target0, &arr0.values.borrow());

        let arr1 = arr.slice(1, 0);
        let target1: Vec<f32> = vec![1.0, 2.0, 5.0, 6.0];
        compare_slices(&target1, &arr1.values.borrow());

        let arr2 = arr.slice(2, 0);
        let target2: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0];
        compare_slices(&target2, &arr2.values.borrow());
    }

    #[test]
    fn reduce_sum() {
        let shape = Shape::new(vec![2, 2, 2]);
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape);

        let arr0 = arr.reduce_sum(0);
        let target0: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];
        compare_slices(&target0, &arr0.values.borrow());

        let arr1 = arr.reduce_sum(1);
        let target1: Vec<f32> = vec![4.0, 6.0, 12.0, 14.0];
        compare_slices(&target1, &arr1.values.borrow());

        let arr2 = arr.reduce_sum(2);
        let target2: Vec<f32> = vec![3.0, 7.0, 11.0, 15.0];
        compare_slices(&target2, &arr2.values.borrow());
    }

    #[test]
    fn reduce_max() {
        let shape = Shape::new(vec![2, 2, 2]);
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape);

        let arr0 = arr.reduce_max(0);
        let target0: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        compare_slices(&target0, &arr0.values.borrow());

        let arr1 = arr.reduce_max(1);
        let target1: Vec<f32> = vec![3.0, 4.0, 7.0, 8.0];
        compare_slices(&target1, &arr1.values.borrow());

        let arr2 = arr.reduce_max(2);
        let target2: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
        compare_slices(&target2, &arr2.values.borrow());
    }

    #[test]
    fn unsqueeze() {
        let arr = Array::<f32>::new(vec![2.0; 6], Shape::new(vec![6]));
        let arr = arr.unsqueeze(0);
        let target = [1, 6];
        compare_slices(&target, &arr.shape.borrow().dimensions);

        let arr = Array::<f32>::new(vec![2.0; 6], Shape::new(vec![6]));
        let arr = arr.unsqueeze(1);
        let target = vec![6, 1];
        compare_slices(&target, &arr.shape.borrow().dimensions);
    }

    #[test]
    fn squeeze() {
        let arr = Array::<f32>::new(vec![2.0; 6], Shape::new(vec![1, 6]));
        let arr = arr.squeeze(0);
        let target = vec![6];
        compare_slices(&target, &arr.shape.borrow().dimensions);

        let arr = Array::<f32>::new(vec![2.0; 6], Shape::new(vec![6, 1]));
        let arr = arr.squeeze(1);
        let target = vec![6];
        compare_slices(&target, &arr.shape.borrow().dimensions);
    }

    #[test]
    fn reshape() {
        let arr = Array::<f32>::new(vec![2.0; 24], Shape::new(vec![2, 3, 4]));
        let arr = arr.reshape(Shape::new(vec![2, 3, 2, 2]));
        let target = vec![2, 3, 2, 2];
        compare_slices(&target, &arr.shape.borrow().dimensions);
    }

    #[test]
    fn matmul() {
        let l_arr = Array::<f32>::new(vec![2.0; 6], Shape::new(vec![2, 3]));
        let r_arr = Array::<f32>::new(vec![3.0; 12], Shape::new(vec![3, 4]));
        let res = l_arr.matmul(&r_arr);
        let target_values = vec![18.0; 8];
        compare_slices(&target_values, &res.values.borrow());
        let target_shape = vec![2, 4];
        compare_slices(&target_shape, &res.shape.borrow().dimensions);
    }

    #[test]
    fn permute() {
        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let new_values = values.permute(&[1, 0]);
        let target = vec![1.0, 3.0, 2.0, 4.0];
        compare_slices(&target, &new_values.values.borrow());

        let values = Array::<f32>::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new(vec![2, 2, 2]),
        );
        let new_values = values.permute(&[1, 0, 2]);
        let target = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
        compare_slices(&target, &new_values.values.borrow());

        let values = Array::<f32>::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new(vec![2, 2, 2]),
        );
        let new_values = values.permute(&[2, 1, 0]);
        let target = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
        compare_slices(&target, &new_values.values.borrow());

        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let new_values = values.permute(&[1, 0]);
        let target = vec![1.0, 3.0, 2.0, 4.0];
        compare_slices(&target, &new_values.values.borrow());

        let values = Array::<f32>::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![1, 2, 3]),
        );
        let new_values = values.permute(&[1, 2, 0]);
        let target = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        compare_slices(&target, &new_values.values.borrow());
    }

    #[test]
    fn padding() {
        let values = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let padding = vec![PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let padded_values = values.pad(padding);
        let target = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        compare_slices(&padded_values.values.borrow(), &target);

        let values = Array::<f32>::new(vec![1.0], Shape::new(vec![1, 1, 1]));
        let padding = vec![PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let padded_values = values.pad(padding);
        let target = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        compare_slices(&padded_values.values.borrow(), &target);
    }

    #[test]
    fn shape_strides() {
        let shape = Shape::new(vec![2, 2, 2]);
        let target_stride = [4, 2, 1];
        assert_eq!(shape.strides, target_stride);
    }

    #[test]
    fn padding_utilities() {
        let padding_axes = vec![PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let shape = Shape::new(vec![2, 2]);
        let padding_helper = Padding::new(padding_axes, &shape);
        assert_eq!(padding_helper.padded_strides, [4, 1]);

        let padding_axes = vec![PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let shape = Shape::new(vec![1, 1, 1]);
        let padding_helper = Padding::new(padding_axes, &shape);
        assert_eq!(padding_helper.padded_strides, [9, 3, 1]);
    }

    #[test]
    fn linear_to_array_index() {
        let shape = Shape::new(vec![2, 2, 2]);

        let res = shape.linear_to_array_index(0);
        let target = [0, 0, 0];
        assert_eq!(res, target);

        let res = shape.linear_to_array_index(1);
        let target = [0, 0, 1];
        assert_eq!(res, target);

        let res = shape.linear_to_array_index(2);
        let target = [0, 1, 0];
        assert_eq!(res, target);

        let res = shape.linear_to_array_index(3);
        let target = [0, 1, 1];
        assert_eq!(res, target);

        let res = shape.linear_to_array_index(4);
        let target = [1, 0, 0];
        assert_eq!(res, target);

        let res = shape.linear_to_array_index(5);
        let target = [1, 0, 1];
        assert_eq!(res, target);

        let res = shape.linear_to_array_index(6);
        let target = [1, 1, 0];
        assert_eq!(res, target);

        let res = shape.linear_to_array_index(7);
        let target = [1, 1, 1];
        assert_eq!(res, target);
    }

    #[test]
    fn array_to_linear_index() {
        let shape = Shape::new(vec![1, 2, 3]);

        let res = shape.array_to_linear_index(&[0, 0, 0]);
        assert_eq!(res, 0);

        let res = shape.array_to_linear_index(&[0, 0, 1]);
        assert_eq!(res, 1);

        let res = shape.array_to_linear_index(&[0, 0, 2]);
        assert_eq!(res, 2);

        let res = shape.array_to_linear_index(&[0, 1, 0]);
        assert_eq!(res, 3);

        let res = shape.array_to_linear_index(&[0, 1, 1]);
        assert_eq!(res, 4);

        let res = shape.array_to_linear_index(&[0, 1, 2]);
        assert_eq!(res, 5);
    }
}
