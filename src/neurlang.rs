use crate::utils::{product, rolling_dimensions_lengths};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum MemoryLayout {
    ColumnMajor,
    RowMajor,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ArrayIndex<const N: usize> {
    pub index: [usize; N],
}
impl<const N: usize> ArrayIndex<N> {
    pub fn new(index: [usize; N]) -> Self {
        ArrayIndex { index }
    }
}

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

/// Prefix padding count, Suffix padding count, Padding value
#[derive(Debug, Clone)]
pub struct PadAxis<T>(pub usize, pub usize, pub T);
pub struct Padding<T, const N: usize> {
    pub axes_padding: [PadAxis<T>; N],
    pub padded_sizes: [usize; N],
    pub padded_strides: [usize; N],
    pub original_strides: [usize; N],
}
impl<T: Clone + Copy, const N: usize> Padding<T, N> {
    pub fn new(axes_padding: [PadAxis<T>; N], shape: &Shape<N>) -> Self {
        let mut padded_sizes = shape.dimensions.clone();
        padded_sizes.iter_mut().zip(axes_padding.iter()).for_each(
            |(size, PadAxis(prefix_count, suffix_count, _))| {
                *size = *size + prefix_count + suffix_count
            },
        );
        let mut padded_strides = [1; N];
        for idx in 1..N {
            padded_strides[idx - 1] = padded_sizes[idx..].iter().fold(1, |res, &val| res * val);
        }
        Padding {
            axes_padding,
            padded_sizes,
            padded_strides,
            original_strides: shape.strides,
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
        if axis_index < N - 1 {
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
pub struct Shape<const N: usize> {
    pub dimensions: [usize; N],
    pub strides: [usize; N],
}
impl<const N: usize> Shape<N> {
    pub fn new(dimensions: [usize; N]) -> Self {
        let strides = rolling_dimensions_lengths(&dimensions);
        Shape {
            dimensions,
            strides,
        }
    }

    pub fn remove(&self, index: ReduceAxis) -> Shape<{ N - 1 }>
    where
        [usize; N - 1]: Sized,
    {
        let mut new_dimensions = [0; { N - 1 }];
        new_dimensions[0..index].copy_from_slice(&self.dimensions[0..index]);
        new_dimensions[index..].copy_from_slice(&self.dimensions[(index + 1)..]);
        Shape::new(new_dimensions)
    }

    pub fn insert(&self, new_axis: NewAxis) -> Shape<{ N + 1 }>
    where
        [usize; N + 1]: Sized,
    {
        let mut new_dimensions = [0; { N + 1 }];
        new_dimensions[0..new_axis.index].copy_from_slice(&self.dimensions[0..new_axis.index]);
        new_dimensions[new_axis.index] = new_axis.axis_size;
        new_dimensions[(new_axis.index + 1)..].copy_from_slice(&self.dimensions[new_axis.index..]);
        Shape::new(new_dimensions)
    }

    pub fn permute(&self, new_order: [usize; N]) -> Self {
        let mut new_shape = [0; N];
        for (permuted_idx, &original_idx) in new_order.iter().enumerate() {
            new_shape[permuted_idx] = self.dimensions[original_idx];
        }
        Self::new(new_shape)
    }

    pub fn len(&self) -> usize {
        N
    }

    pub fn nelem(&self) -> usize {
        product(&self.dimensions)
    }

    pub fn make_ordered_index_array(
        permuted_index: &[usize; N],
        permutation_order: &[usize; N],
    ) -> [usize; N] {
        let mut res = [0; N];
        for (&idx, &value) in permutation_order.iter().zip(permuted_index.iter()) {
            res[idx] = value;
        }
        return res;
    }

    pub fn permute_index_array(
        ordered_index: &[usize; N],
        permutation_order: &[usize; N],
    ) -> [usize; N] {
        let mut res = [0; N];
        for (&idx, &value) in permutation_order.iter().zip(ordered_index.iter()) {
            res[idx] = value;
        }
        return res;
    }

    pub fn linear_index_to_array_index(
        linear_index: usize,
        linear_axes_sizes: &[usize; N],
    ) -> [usize; N] {
        let mut results = [1; N];
        let mut counter = linear_index;
        for (idx, &size) in linear_axes_sizes.iter().enumerate() {
            results[idx] = counter / size;
            counter = counter % size;
        }
        results
    }

    pub fn array_index_to_linear_index(array_index: &[usize; N], axes_sizes: &[usize; N]) -> usize {
        array_index
            .iter()
            .zip(axes_sizes.iter())
            .fold(0, |res, (&idx, &size)| res + (idx * size))
    }
}

pub type ReduceAxis = usize;

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Max,
}

#[derive(Debug)]
pub enum ASTOp<T: ExecuteAST, const N: usize> {
    // Leaf
    Value {
        value: T,
    },

    // Unary
    Negate {
        value: Rc<ASTNode<T, N>>,
    },
    Exponential {
        value: Rc<ASTNode<T, N>>,
    },
    Log {
        value: Rc<ASTNode<T, N>>,
    },

    // Binary
    Add {
        left_value: Rc<ASTNode<T, N>>,
        right_value: Rc<ASTNode<T, N>>,
    },
    Sub {
        left_value: Rc<ASTNode<T, N>>,
        right_value: Rc<ASTNode<T, N>>,
    },
    Mul {
        left_value: Rc<ASTNode<T, N>>,
        right_value: Rc<ASTNode<T, N>>,
    },
    Div {
        left_value: Rc<ASTNode<T, N>>,
        right_value: Rc<ASTNode<T, N>>,
    },
    Pow {
        left_value: Rc<ASTNode<T, N>>,
        right_value: Rc<ASTNode<T, N>>,
    },
    CompareEqual {
        left_value: Rc<ASTNode<T, N>>,
        right_value: Rc<ASTNode<T, N>>,
    },
    Max {
        left_value: Rc<ASTNode<T, N>>,
        right_value: Rc<ASTNode<T, N>>,
    },

    // Reduce
    Reduce {
        value: Rc<ASTNode<T, N>>,
        dim: ReduceAxis,
        op: ReduceOp,
    },

    // Movement ops
    Unsqueeze {
        value: Rc<ASTNode<T, N>>,
        dim: usize,
    },
    Squeeze {
        value: Rc<ASTNode<T, N>>,
        dim: usize,
    },
    Reshape {
        value: Rc<ASTNode<T, N>>,
        new_shape: Shape<N>,
    },
    Permute {
        value: Rc<ASTNode<T, N>>,
        dim_order: Vec<usize>,
    },
    // Pad {
    //     value: Rc<ASTNode<T, N>>,
    //     dim: usize,
    //     pad_value: T,
    // },
    // Stride {
    //     value: Rc<ASTNode<T, N>>,
    //     dim: usize,
    //     stride_value: usize,
    // }
}

#[derive(Debug)]
pub struct ASTNode<T: ExecuteAST, const N: usize> {
    op: ASTOp<T, N>,
    shape: Shape<N>,
}

impl<T: ExecuteAST, const N: usize> ASTNode<T, N>
where
    [usize; N - 1]: Sized,
{
    // Unary
    pub fn negate(self: Rc<Self>) -> Rc<ASTNode<T, N>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Negate { value: self },
            shape: new_shape,
        })
    }
    pub fn exp(self: Rc<Self>) -> Rc<ASTNode<T, N>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Exponential { value: self },
            shape: new_shape,
        })
    }
    pub fn log(self: Rc<Self>) -> Rc<ASTNode<T, N>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Log { value: self },
            shape: new_shape,
        })
    }

    // Binary
    pub fn add(self: Rc<Self>, right_value: Rc<ASTNode<T, N>>) -> Rc<ASTNode<T, N>> {
        assert!(
            self.shape.dimensions == right_value.shape.dimensions,
            "Left tensor (shape: {:?}) and right tensor (shape: {:?}) not of equal shape",
            self.shape,
            right_value.shape
        );

        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Add {
                left_value: self,
                right_value: right_value,
            },
            shape: new_shape,
        })
    }
    pub fn subtract(self: Rc<Self>, right_value: Rc<ASTNode<T, N>>) -> Rc<ASTNode<T, N>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Add {
                left_value: self,
                right_value: right_value.negate(),
            },
            shape: new_shape,
        })
    }

    // // Reduce
    // pub fn reduce(self: Rc<Self>, dim: ReduceAxis, op: ReduceOp) -> Rc<ASTNode<T, {N - 1}>> {
    //     assert!(
    //         self.shape.len() >= dim,
    //         "Axis {} not in tensor of dimensions {}",
    //         dim,
    //         self.shape.len()
    //     );

    //     let new_shape = self.shape.remove(dim);
    //     Rc::new(ASTNode {
    //         op: ASTOp::Reduce {
    //             value: self,
    //             dim: dim,
    //             op: op,
    //         },
    //         shape: new_shape,
    //     })
    // }

    // pub fn unsqueeze(self: Rc<Self>, dim: usize) -> Rc<ASTNode<T, N>> {
    //     assert!(
    //         dim <= self.shape.dimensions.len(),
    //         "Expanded dimension larger than existing shape",
    //     );

    //     let new_shape = self.shape.insert(NewAxis::new(dim, 1));
    //     Rc::new(ASTNode {
    //         op: ASTOp::Unsqueeze {
    //             value: self,
    //             dim: dim,
    //         },
    //         shape: new_shape,
    //     })
    // }
    // pub fn squeeze(self: Rc<Self>, dim: usize) -> Rc<ASTNode<T, N>> {
    //     assert!(
    //         dim <= self.shape.dimensions.len(),
    //         "Expanded dimension larger than existing shape",
    //     );
    //     assert!(
    //         self.shape.dimensions[dim] == 1,
    //         "Dimension to be squeezed is not 1",
    //     );

    //     let new_shape = self.shape.remove(dim);
    //     Rc::new(ASTNode {
    //         op: ASTOp::Squeeze {
    //             value: self,
    //             dim: dim,
    //         },
    //         shape: new_shape,
    //     })
    // }

    pub fn reshape(self: Rc<Self>, new_shape: [usize; N]) -> Rc<ASTNode<T, N>> {
        assert!(
            product(&new_shape) == self.shape.nelem(),
            "Reshaped dimensions number elements ({}) does not match number elements of array ({})",
            product(&new_shape),
            self.shape.nelem(),
        );

        let new_shape = Shape::new(new_shape);
        Rc::new(ASTNode {
            op: ASTOp::Reshape {
                value: self,
                new_shape: new_shape.clone(),
            },
            shape: new_shape,
        })
    }

    // Utils
    pub fn new(value: T, shape: Shape<N>) -> Rc<ASTNode<T, N>> {
        Rc::new(ASTNode {
            op: ASTOp::Value { value },
            shape: shape,
        })
    }
    pub fn execute(&self) -> T {
        match &self.op {
            ASTOp::Value { value } => value.value_v(),
            ASTOp::Negate { value } => value.execute().negate_v(),
            ASTOp::Exponential { value } => value.execute().exp_v(),
            ASTOp::Log { value } => value.execute().log_v(),

            ASTOp::Add {
                left_value,
                right_value,
            } => left_value.execute().add_v(&right_value.execute()),
            ASTOp::Sub {
                left_value,
                right_value,
            } => left_value.execute().sub_v(&right_value.execute()),
            ASTOp::Mul {
                left_value,
                right_value,
            } => left_value.execute().mul_v(&right_value.execute()),
            ASTOp::Div {
                left_value,
                right_value,
            } => left_value.execute().div_v(&right_value.execute()),
            ASTOp::Pow {
                left_value,
                right_value,
            } => left_value.execute().pow_v(&right_value.execute()),
            ASTOp::CompareEqual {
                left_value,
                right_value,
            } => left_value.execute().compare_equal_v(&right_value.execute()),
            ASTOp::Max {
                left_value,
                right_value,
            } => left_value.execute().max_v(&right_value.execute()),

            ASTOp::Reduce { value, dim, op } => value.execute().reduce_v(*dim, *op),
            ASTOp::Unsqueeze { value, dim } => value.execute().unsqueeze_v(*dim),
            ASTOp::Squeeze { value, dim } => value.execute().squeeze_v(*dim),
            ASTOp::Reshape { value, new_shape } => value.execute().reshape_v(new_shape.clone()),
            ASTOp::Permute { value, dim_order } => value.execute().permute_v(dim_order),
        }
    }
}

pub trait ExecuteAST {
    // Leaf
    fn value_v(&self) -> Self;

    // Unary
    fn negate_v(&self) -> Self;
    fn exp_v(&self) -> Self;
    fn log_v(&self) -> Self;

    // Binary
    fn add_v(&self, right_value: &Self) -> Self;
    fn sub_v(&self, right_value: &Self) -> Self;
    fn mul_v(&self, right_value: &Self) -> Self;
    fn div_v(&self, right_value: &Self) -> Self;
    fn pow_v(&self, right_value: &Self) -> Self;
    fn compare_equal_v(&self, right_value: &Self) -> Self;
    fn max_v(&self, right_value: &Self) -> Self;

    // Reduce
    fn reduce_v(&self, axis: ReduceAxis, op: ReduceOp) -> Self;

    // // Movement ops
    fn unsqueeze_v(&self, dim: usize) -> Self;
    fn squeeze_v(&self, dim: usize) -> Self;
    fn reshape_v<const N: usize>(&self, new_shape: Shape<N>) -> Self;
    fn permute_v(&self, axis_ordering: &[usize]) -> Self;
    // fn pad_v(&self, ...) -> Self;
    // fn stride_v(&self, ...) -> Self;

    // // Maybe want to include this? Does make for a way nicer experience
    // fn tensordot_v(&self, right_value: &Self, left_axes: &[usize], right_axes: &[usize]) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_to_array_index() {
        let linear_axes_lengths = [4, 2, 1];

        let res = Shape::linear_index_to_array_index(0, &linear_axes_lengths);
        let target = [0, 0, 0];
        assert_eq!(res, target);

        let res = Shape::linear_index_to_array_index(1, &linear_axes_lengths);
        let target = [0, 0, 1];
        assert_eq!(res, target);

        let res = Shape::linear_index_to_array_index(2, &linear_axes_lengths);
        let target = [0, 1, 0];
        assert_eq!(res, target);

        let res = Shape::linear_index_to_array_index(3, &linear_axes_lengths);
        let target = [0, 1, 1];
        assert_eq!(res, target);

        let res = Shape::linear_index_to_array_index(4, &linear_axes_lengths);
        let target = [1, 0, 0];
        assert_eq!(res, target);

        let res = Shape::linear_index_to_array_index(5, &linear_axes_lengths);
        let target = [1, 0, 1];
        assert_eq!(res, target);

        let res = Shape::linear_index_to_array_index(6, &linear_axes_lengths);
        let target = [1, 1, 0];
        assert_eq!(res, target);

        let res = Shape::linear_index_to_array_index(7, &linear_axes_lengths);
        let target = [1, 1, 1];
        assert_eq!(res, target);
    }

    #[test]
    fn shape_strides() {
        let shape = Shape::new([2, 2, 2]);
        let target_stride = [4, 2, 1];
        assert_eq!(shape.strides, target_stride);
    }

    #[test]
    fn padding_utilities() {
        let padding_axes = [PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let shape = Shape::new([2, 2]);
        let padding_helper = Padding::new(padding_axes, &shape);
        assert_eq!(padding_helper.padded_strides, [4, 1]);

        let padding_axes = [PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0), PadAxis(1, 1, 0.0)];
        let shape = Shape::new([1, 1, 1]);
        let padding_helper = Padding::new(padding_axes, &shape);
        assert_eq!(padding_helper.padded_strides, [9, 3, 1]);
    }
}
