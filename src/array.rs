use crate::indexing::*;
use crate::neurlang::{
    ExecuteAST, MemoryLayout, NewAxis, PadAxis, Padding, ReduceAxis, ReduceOp, Shape,
};
use crate::utils::{calculate_strides, permute};

use num::Float;
use rand::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Array<T, const N: usize>
where
    T: Float,
{
    pub values: Rc<RefCell<Vec<T>>>,
    pub shape: RefCell<Shape<N>>,
    pub layout: MemoryLayout,
}

pub fn rand_f32<const N: usize>(shape: Shape<N>) -> Array<f32, N> {
    let mut rng = rand::thread_rng();
    let total_elems = shape.nelem();
    let values = (0..total_elems).map(|_| rng.gen()).collect::<Vec<_>>();
    Array::new(values, shape)
}

impl<T, const N: usize> Array<T, N>
where
    T: Float + std::fmt::Debug + Default,
{
    // Utils
    pub fn new(values: Vec<T>, shape: Shape<N>) -> Self {
        Array {
            values: Rc::new(RefCell::new(values)),
            shape: RefCell::new(shape),
            layout: MemoryLayout::RowMajor,
        }
    }
    pub fn reference_values(values: Rc<RefCell<Vec<T>>>, shape: Shape<N>) -> Self {
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
    pub fn multiply(&self, right_array: &Self) -> Self {
        self.binary_op(right_array, |(&lval, &rval)| lval * rval)
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

    pub fn slice(&self, axis: usize, index: usize) -> Array<T, { N - 1 }> {
        let res_shape = self.shape.borrow().remove(axis);
        let res_values = self.slice_vector(axis, index);
        Array::new(res_values, res_shape)
    }

    pub fn reduce<F>(&self, axis: usize, reduce_f: F) -> Array<T, { N - 1 }>
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

    pub fn reduce_sum(&self, axis: usize) -> Array<T, { N - 1 }> {
        self.reduce(axis, |(res_val, src_val)| *res_val = *res_val + *src_val)
    }
    pub fn reduce_max(&self, axis: usize) -> Array<T, { N - 1 }> {
        self.reduce(axis, |(res_val, src_val)| {
            *res_val = if *res_val > *src_val {
                *res_val
            } else {
                *src_val
            }
        })
    }

    // Movement ops
    pub fn unsqueeze(&self, axis: usize) -> Array<T, { N + 1 }> {
        let new_shape = self.shape.borrow().insert(NewAxis::new(axis, 1));
        Array::reference_values(self.values.clone(), new_shape)
    }
    pub fn squeeze(&self, axis: usize) -> Array<T, { N - 1 }> {
        let new_shape = self.shape.borrow().remove(axis);
        Array::reference_values(self.values.clone(), new_shape)
    }

    pub fn reshape<const M: usize>(&self, new_shape: Shape<M>) -> Array<T, M> {
        Array::<T, M>::reference_values(self.values.clone(), new_shape)
    }

    fn collect_slice_iterator(&self, axis: usize, index: usize) -> Vec<T> {
        let array = self.values.borrow();
        let slice_iter = make_slice(&self.shape.borrow(), axis, index).into_iter();
        let mut res_values = Vec::with_capacity(slice_iter.n_prefix * slice_iter.n_suffix);
        for (start_idx, end_idx) in slice_iter {
            res_values.extend_from_slice(&array[start_idx..end_idx]);
        }

        res_values
    }

    pub fn permute(&self, permutation: [usize; N]) -> Self {
        let cur_values = self.values.borrow();
        let shape = self.shape.borrow();
        let permuted_shape = shape.permute(&permutation);

        let mut results = vec![Default::default(); cur_values.len()];
        for (idx, &value) in cur_values.iter().enumerate() {
            let ordered_array_index = shape.linear_to_array_index(idx);
            let permuted_array_index = permute(&ordered_array_index, &permutation);
            let permuted_linear_index = permuted_shape.array_to_linear_index(&permuted_array_index);
            results[permuted_linear_index] = value;
        }

        Self::new(results, permuted_shape)
    }

    pub fn pad(&self, axes_padding: [PadAxis<T>; N]) -> Self {
        let padding_helper = Padding::new(axes_padding, &self.shape.borrow());
        let new_nelem = padding_helper.padded_sizes.iter().product();
        let mut new_values = Vec::with_capacity(new_nelem);
        padding_helper.pad_array(&mut new_values, &self.values.borrow(), 0);
        Array::new(new_values, Shape::new(padding_helper.padded_sizes))
    }

    // Higher order ops
    pub fn matmul(&self, right_array: &Array<T, N>) -> Self {
        self.clone()
    }
}

// impl<T> ExecuteAST for Array<T>
// where
//     T: Float + std::fmt::Debug,
// {
//     fn value_v(&self) -> Self {
//         self.clone()
//     }
//     fn negate_v(&self) -> Self {
//         self.negate()
//     }
//     fn exp_v(&self) -> Self {
//         self.exp()
//     }
//     fn log_v(&self) -> Self {
//         self.ln()
//     }
//     fn add_v(&self, right_value: &Self) -> Self {
//         self.add(right_value)
//     }
//     fn max_v(&self, right_value: &Self) -> Self {
//         self.max(right_value)
//     }
//     fn reduce_v(&self, axis: ReduceAxis, op: ReduceOp) -> Self {
//         match op {
//             ReduceOp::Sum => self.reduce_sum(axis),
//             ReduceOp::Max => self.reduce_max(axis),
//         }
//     }
//     fn unsqueeze_v(&self, dim: usize) -> Self {
//         self.unsqueeze(dim)
//     }
//     fn squeeze_v(&self, dim: usize) -> Self {
//         self.squeeze(dim)
//     }
//     fn reshape_v(&self, new_shape: Shape) -> Self {
//         self.reshape(new_shape)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_slices<T: PartialEq>(target: &[T], values: &[T]) {
        let compared = target
            .iter()
            .zip(values.iter())
            .filter(|(targ, val)| targ.eq(val))
            .count();
        assert_eq!(compared, target.len());
    }

    #[test]
    fn test_rolling_dimensions_lengths() {
        let shape = [2, 2, 2];
        let lengths = calculate_strides(&shape);
        let target = [4, 2, 1];
        assert_eq!(target, lengths);
    }

    #[test]
    fn negate() {
        let target: Vec<f32> = vec![-1.0, -2.0, -3.0];
        let shape = Shape::new([3]);
        let arr1 = Array::<f32, 1>::new(vec![1.0, 2.0, 3.0], shape).negate();
        compare_slices(&target, &arr1.values.borrow());
    }

    #[test]
    fn add() {
        let target: Vec<f32> = vec![5.0, 7.0, 9.0];
        let shape = Shape::new([3]);
        let arr1 = Array::<f32, 1>::new(vec![1.0, 2.0, 3.0], shape.clone());
        let arr2 = Array::<f32, 1>::new(vec![4.0, 5.0, 6.0], shape);
        let arr3 = arr1.add(&arr2);
        compare_slices(&target, &arr3.values.borrow());
    }

    #[test]
    fn slice() {
        let shape = Shape::new([2, 2, 2]);
        let arr = Array::<f32, 3>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape);

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
        let shape = Shape::new([2, 2, 2]);
        let arr = Array::<f32, 3>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape);

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
        let shape = Shape::new([2, 2, 2]);
        let arr = Array::<f32, 3>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape);

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
        let arr = Array::<f32, 1>::new(vec![2.0; 6], Shape::new([6]));
        let arr = arr.unsqueeze(0);
        let target = [1, 6];
        compare_slices(&target, &arr.shape.borrow().dimensions);

        let arr = Array::<f32, 1>::new(vec![2.0; 6], Shape::new([6]));
        let arr = arr.unsqueeze(1);
        let target = vec![6, 1];
        compare_slices(&target, &arr.shape.borrow().dimensions);
    }

    #[test]
    fn squeeze() {
        let arr = Array::<f32, 2>::new(vec![2.0; 6], Shape::new([1, 6]));
        let arr = arr.squeeze(0);
        let target = vec![6];
        compare_slices(&target, &arr.shape.borrow().dimensions);

        let arr = Array::<f32, 2>::new(vec![2.0; 6], Shape::new([6, 1]));
        let arr = arr.squeeze(1);
        let target = vec![6];
        compare_slices(&target, &arr.shape.borrow().dimensions);
    }

    #[test]
    fn reshape() {
        let arr = Array::<f32, 3>::new(vec![2.0; 24], Shape::new([2, 3, 4]));
        let arr = arr.reshape(Shape::new([2, 3, 2, 2]));
        let target = vec![2, 3, 2, 2];
        compare_slices(&target, &arr.shape.borrow().dimensions);
    }

    #[test]
    fn matmul() {
        let l_arr = Array::<f32, 2>::new(vec![2.0; 6], Shape::new([2, 3]));
        let r_arr = Array::<f32, 2>::new(vec![3.0; 12], Shape::new([3, 4]));
        let res = l_arr.matmul(&r_arr);
        let target_values = vec![18.0; 8];
        compare_slices(&target_values, &res.values.borrow());
        let target_shape = vec![2, 4];
        compare_slices(&target_shape, &res.shape.borrow().dimensions);
    }

    #[test]
    fn permute() {
        let values = Array::<f32, 2>::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
        let new_values = values.permute([1, 0]);
        let target = vec![1.0, 3.0, 2.0, 4.0];
        compare_slices(&target, &new_values.values.borrow());

        let values = Array::<f32, 3>::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new([2, 2, 2]),
        );
        let new_values = values.permute([1, 0, 2]);
        let target = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
        compare_slices(&target, &new_values.values.borrow());

        let values = Array::<f32, 3>::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new([2, 2, 2]),
        );
        let new_values = values.permute([2, 1, 0]);
        let target = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
        compare_slices(&target, &new_values.values.borrow());

        let values = Array::<f32, 2>::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
        let new_values = values.permute([1, 0]);
        let target = vec![1.0, 3.0, 2.0, 4.0];
        compare_slices(&target, &new_values.values.borrow());

        let values =
            Array::<f32, 3>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([1, 2, 3]));
        let new_values = values.permute([1, 2, 0]);
        let target = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        compare_slices(&target, &new_values.values.borrow());
    }

    #[test]
    fn padding() {
        let values = Array::<f32, 2>::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
        let padding = [PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let padded_values = values.pad(padding);
        let target = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        compare_slices(&padded_values.values.borrow(), &target);

        let values = Array::<f32, 3>::new(vec![1.0], Shape::new([1, 1, 1]));
        let padding = [PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let padded_values = values.pad(padding);
        let target = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        compare_slices(&padded_values.values.borrow(), &target);
    }
}
