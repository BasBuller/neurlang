use crate::indexing::*;
use crate::neurlang::{ExecuteAST, Index, MemoryLayout, ReduceAxis, ReduceOp, Shape};
use crate::utils::count_elements;

use num::Float;
use rand::prelude::*;
use std::cell::RefCell;

// fn column_major_indices(shape: &Shape) -> Vec<Index> {
//     let indices: Vec<Vec<usize>> = Vec::with_capacity(shape.nelem());

// }

#[derive(Debug, Clone)]
pub struct Array<T>
where
    T: Float,
{
    pub values: RefCell<Vec<T>>,
    pub shape: Shape,
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
    T: Float + std::fmt::Debug,
{
    // Utils
    pub fn new(values: Vec<T>, shape: Shape) -> Self {
        Array {
            values: RefCell::new(values),
            shape: shape,
            layout: MemoryLayout::RowMajor,
        }
    }
    fn dupe(&self, values: Vec<T>) -> Self {
        Array {
            values: RefCell::new(values),
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
        self.dupe(iterated)
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
        self.dupe(res_values)
    }
    pub fn add(&self, right_array: &Self) -> Self {
        self.binary_op(right_array, |(&lval, &rval)| lval + rval)
    }
    pub fn max(&self, right_array: &Self) -> Self {
        self.binary_op(
            right_array,
            |(&lval, &rval)| if lval > rval { lval } else { rval },
        )
    }

    // Axis reducing operations
    fn reduce_shape(&self, axis: usize) -> Shape {
        let new_dimensions = self.shape.dimensions[0..axis]
            .iter()
            .chain(self.shape.dimensions[(axis + 1)..].iter())
            .map(|&val| val)
            .collect::<Vec<_>>();
        Shape::new(new_dimensions)
    }

    fn slice_vector(&self, axis: usize, index: usize) -> Vec<T> {
        let array = self.values.borrow();
        let slice_iter = make_slice(&self.shape, axis, index).into_iter();
        let mut res_values = Vec::with_capacity(slice_iter.n_prefix * slice_iter.n_suffix);
        for (start_idx, end_idx) in slice_iter {
            res_values.extend_from_slice(&array[start_idx..end_idx]);
        }

        res_values
    }

    pub fn slice(&self, axis: usize, index: usize) -> Self {
        let res_shape = self.reduce_shape(axis);
        let res_values = self.slice_vector(axis, index);
        Self::new(res_values, res_shape)
    }

    pub fn reduce<F>(&self, axis: usize, reduce_f: F) -> Self
    where
        F: Fn((&mut T, &T)),
    {
        let n_prefix = count_elements(&self.shape.dimensions[0..axis]);
        let n_axis_suffix = count_elements(&self.shape.dimensions[axis..]);
        let n_suffix = count_elements(&self.shape.dimensions[(axis + 1)..]);
        let array = self.values.borrow();

        let res_shape = self.reduce_shape(axis);
        let mut res_values = self.slice_vector(axis, 0);
        for prefix_idx in 0..n_prefix {
            for index in 1..self.shape.dimensions[axis] {
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
        Self::new(res_values, res_shape)
    }

    pub fn reduce_sum(&self, axis: usize) -> Self {
        self.reduce(axis, |(res_val, src_val)| *res_val = *res_val + *src_val)
    }
    pub fn reduce_max(&self, axis: usize) -> Self {
        self.reduce(axis, |(res_val, src_val)| {
            *res_val = if *res_val > *src_val {
                *res_val
            } else {
                *src_val
            }
        })
    }
}

impl<T> ExecuteAST for Array<T>
where
    T: Float + std::fmt::Debug,
{
    fn value_v(&self) -> Self {
        self.clone()
    }
    fn negate_v(&self) -> Self {
        self.negate()
    }
    fn exp_v(&self) -> Self {
        self.exp()
    }
    fn log_v(&self) -> Self {
        self.ln()
    }
    fn add_v(&self, right_value: &Self) -> Self {
        self.add(right_value)
    }
    fn max_v(&self, right_value: &Self) -> Self {
        self.max(right_value)
    }
    fn reduce_v(&self, axis: ReduceAxis, op: ReduceOp) -> Self {
        match op {
            ReduceOp::Sum => self.reduce_sum(axis),
            ReduceOp::Max => self.reduce_max(axis),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_vecs(target: &Vec<f32>, values: &Vec<f32>) {
        let compared = target
            .iter()
            .zip(values.iter())
            .filter(|(targ, val)| targ.eq(val))
            .count();
        assert_eq!(compared, target.len());
    }

    #[test]
    fn negate() {
        let target: Vec<f32> = vec![-1.0, -2.0, -3.0];
        let shape = Shape::new(vec![3]);
        let arr1 = Array::<f32>::new(vec![1.0, 2.0, 3.0], shape).negate();
        compare_vecs(&target, &arr1.values.borrow());
    }

    #[test]
    fn add() {
        let target: Vec<f32> = vec![5.0, 7.0, 9.0];
        let shape = Shape::new(vec![3]);
        let arr1 = Array::<f32>::new(vec![1.0, 2.0, 3.0], shape.clone());
        let arr2 = Array::<f32>::new(vec![4.0, 5.0, 6.0], shape);
        let arr3 = arr1.add(&arr2);
        compare_vecs(&target, &arr3.values.borrow());
    }

    #[test]
    fn slice() {
        let shape = Shape::new(vec![2, 2, 2]);
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape);

        let arr0 = arr.slice(0, 0);
        let target0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        compare_vecs(&target0, &arr0.values.borrow());

        let arr1 = arr.slice(1, 0);
        let target1: Vec<f32> = vec![1.0, 2.0, 5.0, 6.0];
        compare_vecs(&target1, &arr1.values.borrow());

        let arr2 = arr.slice(2, 0);
        let target2: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0];
        compare_vecs(&target2, &arr2.values.borrow());
    }

    #[test]
    fn reduce_sum() {
        let shape = Shape::new(vec![2, 2, 2]);
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape);

        let arr0 = arr.reduce_sum(0);
        let target0: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];
        compare_vecs(&target0, &arr0.values.borrow());

        let arr1 = arr.reduce_sum(1);
        let target1: Vec<f32> = vec![4.0, 6.0, 12.0, 14.0];
        compare_vecs(&target1, &arr1.values.borrow());

        let arr2 = arr.reduce_sum(2);
        let target2: Vec<f32> = vec![3.0, 7.0, 11.0, 15.0];
        compare_vecs(&target2, &arr2.values.borrow());
    }

    #[test]
    fn reduce_max() {
        let shape = Shape::new(vec![2, 2, 2]);
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape);

        let arr0 = arr.reduce_max(0);
        let target0: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        compare_vecs(&target0, &arr0.values.borrow());

        let arr1 = arr.reduce_max(1);
        let target1: Vec<f32> = vec![3.0, 4.0, 7.0, 8.0];
        compare_vecs(&target1, &arr1.values.borrow());

        let arr2 = arr.reduce_max(2);
        let target2: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
        compare_vecs(&target2, &arr2.values.borrow());
    }
}
