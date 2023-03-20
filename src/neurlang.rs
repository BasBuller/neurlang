use crate::utils::count_elements;
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

#[derive(Debug, Clone)]
pub struct Shape<const N: usize> {
    pub dimensions: [usize; N],
}
impl<const N: usize> Shape<N>
// where
//     [usize; {N - 1}]: Sized,
//     [usize; {N + 1}]: Sized
{
    pub fn new(dimensions: [usize; N]) -> Self {
        Shape { dimensions }
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
        [usize; { N + 1 }]: Sized,
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
        // self.dimensions.len()
    }

    pub fn nelem(&self) -> usize {
        count_elements(&self.dimensions)
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
    // Sub {
    //     left_value: Rc<ASTNode<T, N>>,
    //     right_value: Rc<ASTNode<T, N>>,
    // },
    // Mul {
    //     left_value: Rc<ASTNode<T, N>>,
    //     right_value: Rc<ASTNode<T, N>>,
    // },
    // Div {
    //     left_value: Rc<ASTNode<T, N>>,
    //     right_value: Rc<ASTNode<T, N>>,
    // },
    // Pow {
    //     left_value: Rc<ASTNode<T, N>>,
    //     right_value: Rc<ASTNode<T, N>>,
    // },
    // CompareEqual {
    //     left_value: Rc<ASTNode<T, N>>,
    //     right_value: Rc<ASTNode<T, N>>,
    // },
    // Max {
    //     left_value: Rc<ASTNode<T, N>>,
    //     right_value: Rc<ASTNode<T, N>>,
    // },

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
    // Permute {
    //     value: Rc<ASTNode<T, N>>,
    //     dim_order: Vec<usize>,
    // },
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
            count_elements(&new_shape) == self.shape.nelem(),
            "Reshaped dimensions number elements ({}) does not match number elements of array ({})",
            count_elements(&new_shape),
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
            ASTOp::Reduce { value, dim, op } => value.execute().reduce_v(*dim, *op),
            ASTOp::Unsqueeze { value, dim } => value.execute().unsqueeze_v(*dim),
            ASTOp::Squeeze { value, dim } => value.execute().squeeze_v(*dim),
            ASTOp::Reshape { value, new_shape } => value.execute().reshape_v(new_shape.clone()),
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
    // fn sub_v(&self, right_value: &Self) -> Self;
    // fn mul_v(&self, right_value: &Self) -> Self;
    // fn div_v(&self, right_value: &Self) -> Self;
    // fn pow_v(&self, right_value: &Self) -> Self;
    // fn compare_equal_v(&self, right_value: &Self) -> Self;
    fn max_v(&self, right_value: &Self) -> Self;

    // Reduce
    fn reduce_v(&self, axis: ReduceAxis, op: ReduceOp) -> Self;

    // // Movement ops
    fn unsqueeze_v(&self, dim: usize) -> Self;
    fn squeeze_v(&self, dim: usize) -> Self;
    fn reshape_v<const N: usize>(&self, new_shape: Shape<N>) -> Self;
    // fn permute_v(&self, axis_ordering: &[usize]) -> Self;
    // // fn pad_v(&self, ...) -> Self;
    // // fn shrink_v(&self, ...) -> Self;
    // // fn stride_v(&self, ...) -> Self;

    // // Maybe want to include this? Does make for a way nicer experience
    // fn tensordot_v(&self, right_value: &Self, left_axes: &[usize], right_axes: &[usize]) -> Self;
}
