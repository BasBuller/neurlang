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
pub struct Shape {
    pub dimensions: Vec<usize>,
}
impl Shape {
    pub fn new(dimensions: Vec<usize>) -> Self {
        Shape { dimensions }
    }

    pub fn remove(&self, index: ReduceAxis) -> Shape {
        let mut new_dimensions = self.dimensions.clone();
        new_dimensions.remove(index);
        Self::new(new_dimensions)
    }

    pub fn insert(&self, new_axis: NewAxis) -> Self {
        let mut new_dimensions = self.dimensions.clone();
        new_dimensions.insert(new_axis.index, new_axis.axis_size);
        Self::new(new_dimensions)
    }

    pub fn permute(&self, new_order: &Vec<usize>) -> Self {
        let new_shape = new_order
            .iter()
            .map(|&idx| self.dimensions[idx])
            .collect::<Vec<_>>();
        Self::new(new_shape)
    }

    pub fn len(&self) -> usize {
        self.dimensions.len()
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
pub enum ASTOp<T: ExecuteAST> {
    // Leaf
    Value {
        value: T,
    },

    // Unary
    Negate {
        value: Rc<ASTNode<T>>,
    },
    Exponential {
        value: Rc<ASTNode<T>>,
    },
    Log {
        value: Rc<ASTNode<T>>,
    },

    // Binary
    Add {
        left_value: Rc<ASTNode<T>>,
        right_value: Rc<ASTNode<T>>,
    },
    // Sub {
    //     left_value: Rc<ASTNode<T>>,
    //     right_value: Rc<ASTNode<T>>,
    // },
    // Mul {
    //     left_value: Rc<ASTNode<T>>,
    //     right_value: Rc<ASTNode<T>>,
    // },
    // Div {
    //     left_value: Rc<ASTNode<T>>,
    //     right_value: Rc<ASTNode<T>>,
    // },
    // Pow {
    //     left_value: Rc<ASTNode<T>>,
    //     right_value: Rc<ASTNode<T>>,
    // },
    // CompareEqual {
    //     left_value: Rc<ASTNode<T>>,
    //     right_value: Rc<ASTNode<T>>,
    // },
    // Max {
    //     left_value: Rc<ASTNode<T>>,
    //     right_value: Rc<ASTNode<T>>,
    // },

    // Reduce
    Reduce {
        value: Rc<ASTNode<T>>,
        dim: ReduceAxis,
        op: ReduceOp,
    },

    // Movement ops
    Unsqueeze {
        value: Rc<ASTNode<T>>,
        dim: usize,
    },
    Squeeze {
        value: Rc<ASTNode<T>>,
        dim: usize,
    },
    Reshape {
        value: Rc<ASTNode<T>>,
        new_shape: Shape,
    },
    // Permute {
    //     value: Rc<ASTNode<T>>,
    //     dim_order: Vec<usize>,
    // },
    // Pad {
    //     value: Rc<ASTNode<T>>,
    //     dim: usize,
    //     pad_value: T,
    // },
    // Stride {
    //     value: Rc<ASTNode<T>>,
    //     dim: usize,
    //     stride_value: usize,
    // }
}

#[derive(Debug)]
pub struct ASTNode<T: ExecuteAST> {
    op: ASTOp<T>,
    shape: Shape,
}

impl<T: ExecuteAST> ASTNode<T> {
    // Unary
    pub fn negate(self: Rc<Self>) -> Rc<ASTNode<T>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Negate { value: self },
            shape: new_shape,
        })
    }
    pub fn exp(self: Rc<Self>) -> Rc<ASTNode<T>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Exponential { value: self },
            shape: new_shape,
        })
    }
    pub fn log(self: Rc<Self>) -> Rc<ASTNode<T>> {
        let new_shape = self.shape.clone();
        Rc::new(ASTNode {
            op: ASTOp::Log { value: self },
            shape: new_shape,
        })
    }

    // Binary
    pub fn add(self: Rc<Self>, right_value: Rc<ASTNode<T>>) -> Rc<ASTNode<T>> {
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
    pub fn subtract(self: Rc<Self>, right_value: Rc<ASTNode<T>>) -> Rc<ASTNode<T>> {
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
    pub fn reduce(self: Rc<Self>, dim: ReduceAxis, op: ReduceOp) -> Rc<ASTNode<T>> {
        assert!(
            self.shape.len() >= dim,
            "Axis {} not in tensor of dimensions {}",
            dim,
            self.shape.len()
        );

        let new_shape = self.shape.clone();
        new_shape.remove(dim);
        Rc::new(ASTNode {
            op: ASTOp::Reduce {
                value: self,
                dim: dim,
                op: op,
            },
            shape: new_shape,
        })
    }

    pub fn unsqueeze(self: Rc<Self>, dim: usize) -> Rc<ASTNode<T>> {
        assert!(
            dim <= self.shape.dimensions.len(),
            "Expanded dimension larger than existing shape",
        );

        let mut new_shape = self.shape.clone();
        new_shape.dimensions.insert(dim, 1);
        Rc::new(ASTNode {
            op: ASTOp::Unsqueeze {
                value: self,
                dim: dim,
            },
            shape: new_shape,
        })
    }
    pub fn squeeze(self: Rc<Self>, dim: usize) -> Rc<ASTNode<T>> {
        assert!(
            dim <= self.shape.dimensions.len(),
            "Expanded dimension larger than existing shape",
        );
        assert!(
            self.shape.dimensions[dim] == 1,
            "Dimension to be squeezed is not 1",
        );

        let mut new_shape = self.shape.clone();
        new_shape.dimensions.remove(dim);
        Rc::new(ASTNode {
            op: ASTOp::Squeeze {
                value: self,
                dim: dim,
            },
            shape: new_shape,
        })
    }

    pub fn reshape(self: Rc<Self>, new_shape: Vec<usize>) -> Rc<ASTNode<T>> {
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
    pub fn new(value: T, shape: Shape) -> Rc<ASTNode<T>> {
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
    fn reshape_v(&self, new_shape: Shape) -> Self;
    // fn permute_v(&self, axis_ordering: &[usize]) -> Self;
    // // fn pad_v(&self, ...) -> Self;
    // // fn shrink_v(&self, ...) -> Self;
    // // fn stride_v(&self, ...) -> Self;

    // // Maybe want to include this? Does make for a way nicer experience
    // fn tensordot_v(&self, right_value: &Self, left_axes: &[usize], right_axes: &[usize]) -> Self;
}
