use crate::array::{NewAxis, PadAxis, Shape};
use std::rc::Rc;

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
pub enum ASTOp<F, T: ExecuteAST<F>> {
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
        new_shape: Shape,
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
pub struct ASTNode<F, T: ExecuteAST<F>> {
    op: ASTOp<F, T>,
    shape: Shape,
}

impl<F, T: ExecuteAST<F>> ASTNode<F, T> {
    // Unary
    pub fn unary(self: Rc<Self>, op: UnaryOp) -> Rc<ASTNode<F, T>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Unary {
                value: self,
                op: op,
            },
            shape: new_shape,
        })
    }

    // Binary
    pub fn binary(self: Rc<Self>, right_value: Rc<ASTNode<F, T>>, op: BinaryOp) -> Rc<ASTNode<F, T>> {
        assert!(
            self.shape.dimensions == right_value.shape.dimensions,
            "Left tensor (shape: {:?}) and right tensor (shape: {:?}) not of equal shape",
            self.shape,
            right_value.shape
        );

        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Binary {
                left_value: self,
                right_value: right_value,
                op: op,
            },
            shape: new_shape,
        })
    }

    // Reduce
    pub fn reduce(self: Rc<Self>, dim: ReduceAxis, op: ReduceOp) -> Rc<ASTNode<F, T>> {
        assert!(
            self.shape.len() >= dim,
            "Axis {} not in tensor of dimensions {}",
            dim,
            self.shape.len()
        );

        let new_shape = self.shape.remove(dim);
        Rc::new(ASTNode {
            op: ASTOp::Reduce {
                value: self,
                dim: dim,
                op: op,
            },
            shape: new_shape,
        })
    }

    // Movement ops
    pub fn unsqueeze(self: Rc<Self>, dim: usize) -> Rc<ASTNode<F, T>> {
        assert!(
            dim <= self.shape.dimensions.len(),
            "Expanded dimension larger than existing shape",
        );

        let new_shape = self.shape.insert(NewAxis::new(dim, 1));
        Rc::new(ASTNode {
            op: ASTOp::Unsqueeze {
                value: self,
                dim: dim,
            },
            shape: new_shape,
        })
    }

    pub fn squeeze(self: Rc<Self>, dim: usize) -> Rc<ASTNode<F, T>> {
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
                value: self,
                dim: dim,
            },
            shape: new_shape,
        })
    }

    pub fn reshape(self: Rc<Self>, new_shape: Vec<usize>) -> Rc<ASTNode<F, T>> {
        assert!(
            new_shape.iter().product::<usize>() == self.shape.nelem(),
            "Reshaped dimensions number elements ({}) does not match number elements of array ({})",
            new_shape.iter().product::<usize>(),
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
    pub fn new(value: T, shape: Shape) -> Rc<ASTNode<F, T>> {
        Rc::new(ASTNode {
            op: ASTOp::Value { value },
            shape: shape,
        })
    }
    pub fn execute(&self) -> T {
        match &self.op {
            ASTOp::Value { value } => value.value_v(),
            ASTOp::Unary { value, op } => {
                let value = value.execute();
                match op {
                    UnaryOp::Negate => value.negate_v(),
                    UnaryOp::Exponential => value.exp_v(),
                    UnaryOp::Log => value.log_v(),
                }
            },
            ASTOp::Binary { left_value, right_value, op } => {
                let left_value = left_value.execute();
                let right_value = right_value.execute();
                match op {
                    BinaryOp::Add => left_value.add_v(&right_value),
                    BinaryOp::Sub => left_value.sub_v(&right_value),
                    BinaryOp::Mul => left_value.mul_v(&right_value),
                    BinaryOp::Div => left_value.div_v(&right_value),
                    BinaryOp::Pow => left_value.pow_v(&right_value),
                    BinaryOp::CompareEqual => left_value.compare_equal_v(&right_value),
                    BinaryOp::Max => left_value.max_v(&right_value),
                }
            },
            ASTOp::Reduce { value, dim, op } => {
                let value = value.execute();
                match op {
                    ReduceOp::Sum => value.reduce_sum_v(*dim),
                    ReduceOp::Max => value.reduce_max_v(*dim),
                }
            },
            ASTOp::Unsqueeze { value, dim } => value.execute().unsqueeze_v(*dim),
            ASTOp::Squeeze { value, dim } => value.execute().squeeze_v(*dim),
            ASTOp::Reshape { value, new_shape } => value.execute().reshape_v(new_shape.clone()),
            ASTOp::Permute { value, dim_order } => value.execute().permute_v(dim_order),
            ASTOp::Pad {
                value,
                axes_padding,
            } => value.execute().pad_v(axes_padding),
        }
    }
}

pub trait ExecuteAST<F> {
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
    fn reduce_max_v(&self, axis: ReduceAxis) -> Self;
    fn reduce_sum_v(&self, axis: ReduceAxis) -> Self;

    // // Movement ops
    fn unsqueeze_v(&self, dim: usize) -> Self;
    fn squeeze_v(&self, dim: usize) -> Self;
    fn reshape_v(&self, new_shape: Shape) -> Self;
    fn permute_v(&self, axis_ordering: &[usize]) -> Self;
    fn pad_v(&self, axes_padding: &Vec<PadAxis<F>>) -> Self;
    // fn stride_v(&self, ...) -> Self;

    // // Maybe want to include this? Does make for a way nicer experience
    // fn tensordot_v(&self, right_value: &Self, left_axes: &[usize], right_axes: &[usize]) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
}
