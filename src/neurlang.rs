use crate::utils::{calculate_strides, permute};
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
pub struct PadAxis<T: Clone>(pub usize, pub usize, pub T);
pub struct Padding<T: Clone> {
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
        Shape::new(new_dimensions)
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

    pub fn linear_to_array_index_with_target(
        &self,
        linear_index: usize,
        target_slice: &mut [usize],
    ) {
        let mut counter = linear_index;
        for (target, &size) in target_slice.iter_mut().zip(self.strides.iter()) {
            *target = counter / size;
            counter = counter % size;
        }
    }

    pub fn linear_to_array_index(&self, linear_index: usize) -> Vec<usize> {
        let mut results = vec![0; self.len()];
        let mut counter = linear_index;
        results
            .iter_mut()
            .zip(self.strides.iter())
            .for_each(|(target, &stride)| {
                *target = counter / stride;
                counter = counter % stride;
            });
        results
    }

    pub fn linear_permute_linear(&self, linear_index: usize, permuted_strides: &[usize]) -> usize {
        let mut result = 0;
        let mut counter = linear_index;
        for (&linear_stride, &permuted_stride) in self.strides.iter().zip(permuted_strides.iter()) {
            // Ordered operations first
            let linear_idx = counter / linear_stride;
            counter = counter % linear_stride;

            // Convert to permuted linear
            result += linear_idx * permuted_stride;
        }
        result
    }
    
    pub fn strided(&self, strides: &[usize]) -> Self {
        let mut new_dimensions = self.dimensions.clone();
        for (n_dim, &stride_mul) in new_dimensions.iter_mut().zip(strides.iter()) {
            *n_dim /= stride_mul;
        }

        let mut new_strides = self.strides.clone();
        for (n_stride, &stride_mul) in new_strides.iter_mut().zip(strides.iter()) {
            *n_stride *= stride_mul;
        }
        
        Shape {
            dimensions: new_dimensions,
            strides: new_strides,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MemoryLayout {
    ColumnMajor,
    RowMajor,
}

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Max,
}
pub type ReduceAxis = usize;

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Negate,
    Exponential,
    Log,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    CompareEqual,
    Max,
}

#[derive(Debug)]
pub enum ASTOp<F: Clone, T: ExecuteAST<F>> {
    Value {
        value: T,
    },
    Unary {
        value: Rc<ASTNode<F, T>>,
        op: UnaryOp,
    },
    Binary {
        left_value: Rc<ASTNode<F, T>>,
        right_value: Rc<ASTNode<F, T>>,
        op: BinaryOp,
    },
    Reduce {
        value: Rc<ASTNode<F, T>>,
        dim: ReduceAxis,
        op: ReduceOp,
    },

    // Movement ops, these are separate from the rest because they do not have consistent inputs
    Unsqueeze {
        value: Rc<ASTNode<F, T>>,
        dim: usize,
    },
    Squeeze {
        value: Rc<ASTNode<F, T>>,
        dim: usize,
    },
    Reshape {
        value: Rc<ASTNode<F, T>>,
        new_shape: Vec<usize>,
    },
    Permute {
        value: Rc<ASTNode<F, T>>,
        dim_order: Vec<usize>,
    },
    Pad {
        value: Rc<ASTNode<F, T>>,
        axes_padding: Vec<PadAxis<F>>,
    },
    // Stride {
    //     value: Rc<ASTNode<F, T>>,
    //     dim: usize,
    //     stride_value: usize,
    // }
}

#[derive(Debug)]
pub struct ASTNode<F: Clone, T: ExecuteAST<F>> {
    op: ASTOp<F, T>,
    shape: Shape,
}

impl<F: Clone, T: ExecuteAST<F>> ASTNode<F, T> {
    // Unary
    pub fn unary(self: &Rc<Self>, op: UnaryOp) -> Rc<ASTNode<F, T>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Unary {
                value: self.clone(),
                op: op,
            },
            shape: new_shape,
        })
    }

    // Binary
    pub fn binary(
        self: &Rc<Self>,
        right_value: &Rc<ASTNode<F, T>>,
        op: BinaryOp,
    ) -> Rc<ASTNode<F, T>> {
        assert!(
            self.shape.dimensions == right_value.shape.dimensions,
            "Left tensor (shape: {:?}) and right tensor (shape: {:?}) not of equal shape",
            self.shape,
            right_value.shape
        );

        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Binary {
                left_value: self.clone(),
                right_value: right_value.clone(),
                op: op,
            },
            shape: new_shape,
        })
    }

    // Reduce
    pub fn reduce(self: &Rc<Self>, dim: ReduceAxis, op: ReduceOp) -> Rc<ASTNode<F, T>> {
        assert!(
            self.shape.len() >= dim,
            "Axis {} not in tensor of dimensions {}",
            dim,
            self.shape.len()
        );

        let new_shape = self.shape.remove(dim);
        Rc::new(ASTNode {
            op: ASTOp::Reduce {
                value: self.clone(),
                dim: dim,
                op: op,
            },
            shape: new_shape,
        })
    }

    // Movement ops
    pub fn unsqueeze(self: &Rc<Self>, dim: usize) -> Rc<ASTNode<F, T>> {
        assert!(
            dim <= self.shape.dimensions.len(),
            "Expanded dimension larger than existing shape",
        );

        let new_shape = self.shape.insert(NewAxis::new(dim, 1));
        Rc::new(ASTNode {
            op: ASTOp::Unsqueeze {
                value: self.clone(),
                dim: dim,
            },
            shape: new_shape,
        })
    }

    pub fn squeeze(self: &Rc<Self>, dim: usize) -> Rc<ASTNode<F, T>> {
        assert!(
            dim <= self.shape.dimensions.len(),
            "Expanded dimension larger than existing shape",
        );
        assert!(
            self.shape.dimensions[dim] == 1,
            "Dimension to be squeezed is not 1",
        );

        let new_shape = self.shape.remove(dim);
        Rc::new(ASTNode {
            op: ASTOp::Squeeze {
                value: self.clone(),
                dim: dim,
            },
            shape: new_shape,
        })
    }

    pub fn reshape(self: &Rc<Self>, new_shape_vec: Vec<usize>) -> Rc<ASTNode<F, T>> {
        assert!(
            new_shape_vec.iter().product::<usize>() == self.shape.nelem(),
            "Reshaped dimensions number elements ({}) does not match number elements of array ({})",
            new_shape_vec.iter().product::<usize>(),
            self.shape.nelem(),
        );

        let new_shape = Shape::new(new_shape_vec.clone());
        Rc::new(ASTNode {
            op: ASTOp::Reshape {
                value: self.clone(),
                new_shape: new_shape_vec,
            },
            shape: new_shape,
        })
    }

    // Utils
    pub fn new(value: T, shape: Shape) -> Rc<ASTNode<F, T>> {
        Rc::new(ASTNode {
            op: ASTOp::Value { value },
            shape: shape,
        })
    }
    pub fn execute(&self) -> T {
        match &self.op {
            ASTOp::Value { value } => value.value_ast(),
            ASTOp::Unary { value, op } => {
                let value = value.execute();
                match op {
                    UnaryOp::Negate => value.negate_ast(),
                    UnaryOp::Exponential => value.exp_ast(),
                    UnaryOp::Log => value.log_ast(),
                }
            }
            ASTOp::Binary {
                left_value,
                right_value,
                op,
            } => {
                let left_value = left_value.execute();
                let right_value = right_value.execute();
                match op {
                    BinaryOp::Add => left_value.add_ast(&right_value),
                    BinaryOp::Sub => left_value.sub_ast(&right_value),
                    BinaryOp::Mul => left_value.mul_ast(&right_value),
                    BinaryOp::Div => left_value.div_ast(&right_value),
                    BinaryOp::Pow => left_value.pow_ast(&right_value),
                    BinaryOp::CompareEqual => left_value.compare_equal_ast(&right_value),
                    BinaryOp::Max => left_value.max_ast(&right_value),
                }
            }
            ASTOp::Reduce { value, dim, op } => {
                let array = value.execute();
                match op {
                    ReduceOp::Sum => array.reduce_sum_ast(&value.shape, *dim),
                    ReduceOp::Max => array.reduce_max_ast(&value.shape, *dim),
                }
            }
            ASTOp::Unsqueeze { value, dim } => value.execute().unsqueeze_ast(&value.shape, *dim),
            ASTOp::Squeeze { value, dim } => value.execute().squeeze_ast(&value.shape, *dim),
            ASTOp::Reshape { value, new_shape } => {
                value.execute().reshape_ast(&value.shape, new_shape)
            }
            ASTOp::Permute { value, dim_order } => {
                value.execute().permute_ast(&value.shape, dim_order)
            }
            ASTOp::Pad {
                value,
                axes_padding,
            } => value.execute().pad_ast(&value.shape, (*axes_padding).clone()),
        }
    }
}

pub trait ExecuteAST<F: Clone> {
    // Leaf
    fn value_ast(&self) -> Self;

    // Unary
    fn negate_ast(&self) -> Self;
    fn exp_ast(&self) -> Self;
    fn log_ast(&self) -> Self;

    // Binary
    fn add_ast(&self, right_value: &Self) -> Self;
    fn sub_ast(&self, right_value: &Self) -> Self;
    fn mul_ast(&self, right_value: &Self) -> Self;
    fn div_ast(&self, right_value: &Self) -> Self;
    fn pow_ast(&self, right_value: &Self) -> Self;
    fn compare_equal_ast(&self, right_value: &Self) -> Self;
    fn max_ast(&self, right_value: &Self) -> Self;

    // Reduce
    fn reduce_max_ast(&self, shape: &Shape, axis: ReduceAxis) -> Self;
    fn reduce_sum_ast(&self, shape: &Shape, axis: ReduceAxis) -> Self;

    // // Movement ops
    fn unsqueeze_ast(&self, shape: &Shape, dim: usize) -> Self;
    fn squeeze_ast(&self, shape: &Shape, dim: usize) -> Self;
    fn reshape_ast(&self, shape: &Shape, new_shape: &[usize]) -> Self;
    fn permute_ast(&self, shape: &Shape, axis_ordering: &[usize]) -> Self;
    fn pad_ast(&self, shape: &Shape, axes_padding: Vec<PadAxis<F>>) -> Self;
    // fn stride_v(&self, ...) -> Self;

    // Higher order
    fn matmul_ast(&self, left_shape: &Shape, right_value: &Self, right_shape: &Shape) -> Self;
    // fn tensordot_v(&self, right_value: &Self, left_axes: &[usize], right_axes: &[usize]) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

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
    
    #[test]
    fn test_shape_strides() {
        let shape = Shape::new(vec![512, 512]);
        let strides = [2, 1];
        let strided_shape = shape.strided(&strides);
        let target_stride = [1024, 1];
        assert_eq!(strided_shape.strides, target_stride);
        let target_dimensions = [256, 512];
        assert_eq!(strided_shape.dimensions, target_dimensions);
        
        let strides = [1, 2];
        let strided_shape = shape.strided(&strides);
        let target_stride = [512, 2];
        assert_eq!(strided_shape.strides, target_stride);
        let target_dimensions = [512, 256];
        assert_eq!(strided_shape.dimensions, target_dimensions);

        let shape = Shape::new(vec![2, 2, 2]);
        let target_stride = [4, 2, 1];
        assert_eq!(shape.strides, target_stride);
    }

    #[test]
    fn test_padding_utilities() {
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
    fn test_linear_to_array_index() {
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
    fn test_array_to_linear_index() {
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
