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
        Shape {
            dimensions: new_dimensions,
        }
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
pub struct ASTNode<T: ExecuteAST> {
    op: ASTOp<T>,
    shape: Shape,
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
    // fn expand_v(&self, axis: usize) -> Self;
    // fn reshape_v(&self, new_shape: &Shape) -> Self;
    // fn permute_v(&self, axis_ordering: &[usize]) -> Self;
    // // fn pad_v(&self, ...) -> Self;
    // // fn shrink_v(&self, ...) -> Self;
    // // fn stride_v(&self, ...) -> Self;

    // // Maybe want to include this? Does make for a way nicer experience
    // fn tensordot_v(&self, right_value: &Self, left_axes: &[usize], right_axes: &[usize]) -> Self;
}
