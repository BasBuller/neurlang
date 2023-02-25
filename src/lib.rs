use num::Float;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::rc::Rc;

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

pub type Shape = Vec<usize>;

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
    // // Reduce
    // Reduce {
    //     value: Rc<ASTNode<T>>,
    //     dim: ReduceAxis,
    //     op: ReduceOp,
    // },
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
            self.shape == right_value.shape,
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

    // // Reduce
    // fn reduce(self: Rc<Self>, dim: ReduceAxis, op: ReduceOp) -> Rc<ASTNode<T>> {
    //     assert!(
    //         self.shape.len() >= dim,
    //         "Axis {} not in tensor of dimensions {}",
    //         dim,
    //         self.shape.len()
    //     );

    //     let mut new_shape = self.shape.clone();
    //     new_shape.remove(dim);
    //     Rc::new(ASTNode {
    //         op: ASTOp::Reduce {
    //             value: self,
    //             dim: dim,
    //             op: op,
    //         },
    //         shape: new_shape,
    //     })
    // }

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
            } => left_value.execute().add_v(right_value.execute()),
            // ASTOp::Reduce { value, dim, op } => value.execute().reduce(*dim, *op),
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
    fn add_v(&self, right_value: Self) -> Self;

    // // Reduce
    // fn reduce(&self, dim: ReduceAxis, op: ReduceOp) -> Self;
}

#[derive(Debug)]
pub enum MemoryLayout {
    ColumnMajor,
    RowMajor,
}

#[derive(Debug)]
pub struct Array<T>
where
    T: Float,
{
    pub values: RefCell<Vec<T>>,
    pub shape: Shape,
    pub layout: MemoryLayout,
}

impl<T> Array<T>
where
    T: Float + Copy,
{
    // Utils
    pub fn new(values: Vec<T>, shape: Shape) -> Self {
        Array {
            values: RefCell::new(values),
            shape: shape,
            layout: MemoryLayout::ColumnMajor,
        }
    }
    fn dupe(&self, values: Vec<T>) -> Self {
        Array {
            values: RefCell::new(values),
            shape: self.shape.clone(),
            layout: MemoryLayout::ColumnMajor,
        }
    }
    fn map_values(&self, f: fn(&T) -> T) -> Vec<T> {
        self.values.borrow().iter().map(f).collect::<Vec<_>>()
    }
    fn zip_map_values(&self, right_array: &Self, f: fn((&T, &T)) -> T) -> Vec<T> {
        self.values
            .borrow()
            .iter()
            .zip(right_array.values.borrow().iter())
            .map(f)
            .collect::<Vec<_>>()
    }

    // Unary
    pub fn negate(&self) -> Self {
        let negated = self.map_values(|value| -*value);
        self.dupe(negated)
    }
    pub fn exp(&self) -> Self {
        let exponated = self.map_values(|value| value.exp());
        self.dupe(exponated)
    }
    pub fn ln(&self) -> Self {
        let lns = self.map_values(|value| value.ln());
        self.dupe(lns)
    }

    // Binary
    pub fn add(&self, right_array: &Self) -> Self {
        let added_values = self.zip_map_values(right_array, |(lval, rval)| *lval + *rval);
        self.dupe(added_values)
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
        let arr1 = Array::<f32>::new(vec![1.0, 2.0, 3.0], vec![3]).negate();
        compare_vecs(&target, &arr1.values.borrow());
    }

    #[test]
    fn add() {
        let target: Vec<f32> = vec![5.0, 7.0, 9.0];
        let arr1 = Array::<f32>::new(vec![1.0, 2.0, 3.0], vec![3]);
        let arr2 = Array::<f32>::new(vec![4.0, 5.0, 6.0], vec![3]);
        let arr3 = arr1.add(&arr2);
        compare_vecs(&target, &arr3.values.borrow());
    }
}
