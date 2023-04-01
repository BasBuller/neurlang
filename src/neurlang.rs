use std::rc::Rc;
use crate::array::PadAxis;

#[derive(Debug, Clone)]
pub enum MemoryLayout {
    ColumnMajor,
    RowMajor,
}

fn calculate_strides(dimensions: &[usize]) -> Vec<usize> {
    let mut results = vec![1; dimensions.len()];
    for idx in (0..dimensions.len() - 1).rev() {
        results[idx] = results[idx + 1] * dimensions[idx + 1];
    }
    results
}

#[derive(Debug, Clone)]
pub struct TrackingShape {
    pub dimensions: Vec<usize>,
    pub strides: Vec<usize>,
}
impl TrackingShape {
    pub fn new(dimensions: Vec<usize>) -> Self {
        let strides = calculate_strides(&dimensions);
        Self {
            dimensions,
            strides,
        }
    }

    pub fn init(dimensions: Vec<usize>, strides: Vec<usize>) -> Self {
        Self {
            dimensions,
            strides,
        }
    }
    
    pub fn len(&self) -> usize {
        self.dimensions.len()
    }
    
    pub fn nelem(&self) -> usize {
        self.dimensions.iter().product()
    }
    
    pub fn remove(&self, index: usize) -> Self {
        let mut new_dimensions = self.dimensions.clone();
        new_dimensions.remove(index);
        Self::new(new_dimensions)
    }
    
    pub fn insert(&self, index: usize, size: usize) -> Self {
        let mut new_dimensions = self.dimensions.clone();
        new_dimensions.insert(index, size);
        Self::new(new_dimensions)
    }
}

pub type ReduceAxis = usize;

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Max,
}

#[derive(Debug)]
pub enum ASTOp<F, T: ExecuteAST<F>> {
    // Leaf
    Value {
        value: T,
    },

    // Unary
    Negate {
        value: Rc<ASTNode<F, T>>,
    },
    Exponential {
        value: Rc<ASTNode<F, T>>,
    },
    Log {
        value: Rc<ASTNode<F, T>>,
    },

    // Binary
    Add {
        left_value: Rc<ASTNode<F, T>>,
        right_value: Rc<ASTNode<F, T>>,
    },
    Sub {
        left_value: Rc<ASTNode<F, T>>,
        right_value: Rc<ASTNode<F, T>>,
    },
    Mul {
        left_value: Rc<ASTNode<F, T>>,
        right_value: Rc<ASTNode<F, T>>,
    },
    Div {
        left_value: Rc<ASTNode<F, T>>,
        right_value: Rc<ASTNode<F, T>>,
    },
    Pow {
        left_value: Rc<ASTNode<F, T>>,
        right_value: Rc<ASTNode<F, T>>,
    },
    CompareEqual {
        left_value: Rc<ASTNode<F, T>>,
        right_value: Rc<ASTNode<F, T>>,
    },
    Max {
        left_value: Rc<ASTNode<F, T>>,
        right_value: Rc<ASTNode<F, T>>,
    },

    // Reduce
    Reduce {
        value: Rc<ASTNode<F, T>>,
        dim: ReduceAxis,
        op: ReduceOp,
    },

    // Movement ops
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
        new_shape: TrackingShape,
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
    shape: TrackingShape,
}

impl<F, T: ExecuteAST<F>> ASTNode<F, T> {
    // Unary
    pub fn negate(self: Rc<Self>) -> Rc<ASTNode<F, T>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Negate { value: self },
            shape: new_shape,
        })
    }
    pub fn exp(self: Rc<Self>) -> Rc<ASTNode<F, T>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Exponential { value: self },
            shape: new_shape,
        })
    }
    pub fn log(self: Rc<Self>) -> Rc<ASTNode<F, T>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Log { value: self },
            shape: new_shape,
        })
    }

    // Binary
    pub fn add(self: Rc<Self>, right_value: Rc<ASTNode<F, T>>) -> Rc<ASTNode<F, T>> {
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
    pub fn subtract(self: Rc<Self>, right_value: Rc<ASTNode<F, T>>) -> Rc<ASTNode<F, T>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Add {
                left_value: self,
                right_value: right_value.negate(),
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

    pub fn unsqueeze(self: Rc<Self>, dim: usize) -> Rc<ASTNode<F, T>> {
        assert!(
            dim <= self.shape.dimensions.len(),
            "Expanded dimension larger than existing shape",
        );

        let new_shape = self.shape.insert(dim, 1);
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

        let new_shape = TrackingShape::new(new_shape);
        Rc::new(ASTNode {
            op: ASTOp::Reshape {
                value: self,
                new_shape: new_shape.clone(),
            },
            shape: new_shape,
        })
    }

    // Utils
    pub fn new(value: T, shape: TrackingShape) -> Rc<ASTNode<F, T>> {
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
            ASTOp::Pad { value, axes_padding } => value.execute().pad_v(axes_padding),
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
    fn reduce_v(&self, axis: ReduceAxis, op: ReduceOp) -> Self;

    // // Movement ops
    fn unsqueeze_v(&self, dim: usize) -> Self;
    fn squeeze_v(&self, dim: usize) -> Self;
    fn reshape_v(&self, new_shape: TrackingShape) -> Self;
    fn permute_v(&self, axis_ordering: &[usize]) -> Self;
    fn pad_v(&self, axes_padding: &[PadAxis<F>])-> Self;
    // fn stride_v(&self, ...) -> Self;

    // // Maybe want to include this? Does make for a way nicer experience
    // fn tensordot_v(&self, right_value: &Self, left_axes: &[usize], right_axes: &[usize]) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
}
