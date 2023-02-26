use num::Float;
use rand::prelude::*;
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

#[derive(Debug, Clone)]
pub enum MemoryLayout {
    ColumnMajor,
    RowMajor,
}

#[derive(Debug, Clone)]
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
    T: Float,
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
    pub fn negate(&self) -> Self {
        let negated = self
            .values
            .borrow()
            .iter()
            .map(|&value| -value)
            .collect::<Vec<_>>();
        self.dupe(negated)
    }
    pub fn inpl_negate(&self) {
        self.values
            .borrow_mut()
            .iter_mut()
            .map(|value| *value = -(*value))
            .count();
    }
    pub fn exp(&self) -> Self {
        let exponated = self
            .values
            .borrow()
            .iter()
            .map(|value| value.exp())
            .collect::<Vec<_>>();
        self.dupe(exponated)
    }
    pub fn inpl_exp(&self) {
        self.values
            .borrow_mut()
            .iter_mut()
            .map(|value| *value = value.exp())
            .count();
    }
    pub fn ln(&self) -> Self {
        let lns = self
            .values
            .borrow()
            .iter()
            .map(|value| value.ln())
            .collect::<Vec<_>>();
        self.dupe(lns)
    }
    pub fn inpl_ln(&self) {
        self.values
            .borrow_mut()
            .iter_mut()
            .map(|value| *value = value.ln())
            .count();
    }

    // Binary
    pub fn add(&self, right_array: &Self) -> Self {
        let added_values = self
            .values
            .borrow()
            .iter()
            .zip(right_array.values.borrow().iter())
            .map(|(&lval, &rval)| lval + rval)
            .collect::<Vec<_>>();
        self.dupe(added_values)
    }
    
    // Axis reducing operations
    fn slice(&self, axis: usize, index: usize) -> Self {
        let n_prefix = self.shape[0..axis].iter().fold(1, |res, &value| res * value);
        let n_axis_suffix = self.shape[axis..].iter().fold(1, |res, &value| res * value);
        let n_suffix = self.shape[(axis + 1..)].iter().fold(1, |res, &value| res * value);    
        let array = self.values.borrow();

        let res_shape = self.shape[0..axis].iter().chain(self.shape[(axis + 1)..].iter()).map(|&val| val).collect::<Vec<_>>();
        let mut res_values = Vec::with_capacity(n_prefix * n_suffix);
        for prefix_idx in 0..n_prefix {
            for suffix_idx in 0..n_suffix{
                let arr_idx = (prefix_idx * n_axis_suffix) + (index * n_suffix) + suffix_idx;
                res_values.push(array[arr_idx]);
                // let res_idx = (prefix_idx * n_suffix) + suffix_idx;
                // res_values[res_idx] = array[arr_idx];
            }
        }
        Self::new(res_values, res_shape)
    }
}

pub fn rand_f32(shape: Shape) -> Array<f32> {
    let mut rng = rand::thread_rng();
    let total_elems = shape.iter().fold(1, |res, val| res * (*val));
    let values = (0..total_elems).map(|_| rng.gen()).collect::<Vec<_>>();
    Array::new(values, shape)
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
    
    #[test]
    fn slice() {
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);

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
}
