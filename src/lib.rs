use num::Float;
use rand::prelude::*;
use std::cell::RefCell;
use std::iter::Map;
use std::ops::Range;
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

    // Reduce
    fn reduce(self: Rc<Self>, dim: ReduceAxis, op: ReduceOp) -> Rc<ASTNode<T>> {
        assert!(
            self.shape.len() >= dim,
            "Axis {} not in tensor of dimensions {}",
            dim,
            self.shape.len()
        );

        let mut new_shape = self.shape.clone();
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
            } => left_value.execute().add_v(right_value.execute()),
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
    fn add_v(&self, right_value: Self) -> Self;

    // Reduce
    fn reduce_v(&self, dim: ReduceAxis, op: ReduceOp) -> Self;
}

#[derive(Debug, Clone)]
pub enum MemoryLayout {
    // ColumnMajor,
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

// Hot loops wrapped in a function such that the compiler can optimize them well!
//   TODO: See if I can refactor this into a single generator function or macro that takes only
//   the core functionality and wraps it in the map logic
pub fn negate<T: Float>(values: &[T]) -> Vec<T> {
    values.iter().map(|&value| -value).collect::<Vec<_>>()
}
pub fn inpl_negate<T: Float>(values: &mut [T]) {
    values.iter_mut().map(|value| *value = -(*value)).count();
}
pub fn exp<T: Float>(values: &[T]) -> Vec<T> {
    values.iter().map(|value| value.exp()).collect::<Vec<_>>()
}
pub fn inpl_exp<T: Float>(values: &mut [T]) {
    values.iter_mut().map(|value| *value = value.exp()).count();
}
pub fn ln<T: Float>(values: &[T]) -> Vec<T> {
    values.iter().map(|value| value.ln()).collect::<Vec<_>>()
}
pub fn inpl_ln<T: Float>(values: &mut [T]) {
    values.iter_mut().map(|value| *value = value.ln()).count();
}
pub fn add<T: Float>(lvalues: &[T], rvalues: &[T]) -> Vec<T> {
    lvalues
        .iter()
        .zip(rvalues.iter())
        .map(|(&lval, &rval)| lval + rval)
        .collect::<Vec<_>>()
}
pub fn inpl_add<T: Float>(lvalues: &mut [T], rvalues: &[T]) {
    lvalues
        .iter_mut()
        .zip(rvalues.iter())
        .map(|(lval, &rval)| *lval = (*lval) + rval)
        .count();
}

fn count_elements(shape: &[usize]) -> usize {
    shape.iter().fold(1, |res, &val| res * val)
}

impl<T> Array<T>
where
    T: Float + std::fmt::Debug,
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
        let negated = negate(&self.values.borrow());
        self.dupe(negated)
    }
    pub fn inpl_negate(&self) {
        inpl_negate(&mut self.values.borrow_mut());
    }
    pub fn exp(&self) -> Self {
        let exponated = exp(&self.values.borrow());
        self.dupe(exponated)
    }
    pub fn inpl_exp(&self) {
        inpl_exp(&mut self.values.borrow_mut());
    }
    pub fn ln(&self) -> Self {
        let lns = ln(&self.values.borrow());
        self.dupe(lns)
    }
    pub fn inpl_ln(&self) {
        inpl_ln(&mut self.values.borrow_mut());
    }

    // Binary
    pub fn add(&self, right_array: &Self) -> Self {
        let added_values = add(&self.values.borrow(), &right_array.values.borrow());
        self.dupe(added_values)
    }

    // Axis reducing operations
    fn reduce_shape(&self, axis: usize) -> Shape {
        self.shape[0..axis]
            .iter()
            .chain(self.shape[(axis + 1)..].iter())
            .map(|&val| val)
            .collect::<Vec<_>>()
    }

    fn slice_vector(&self, axis: usize, index: usize) -> Vec<T> {
        let n_prefix = count_elements(&self.shape[0..axis]);
        let n_axis_suffix = count_elements(&self.shape[axis..]);
        let n_suffix = count_elements(&self.shape[(axis + 1)..]);
        let array = self.values.borrow();

        let mut res_values = Vec::with_capacity(n_prefix * n_suffix);
        for prefix_idx in 0..n_prefix {
            let src_start_idx = (prefix_idx * n_axis_suffix) + (index * n_suffix);
            let src_end_idx = src_start_idx + n_suffix;
            res_values.extend_from_slice(&array[src_start_idx..src_end_idx]);
        }

        res_values
    }

    pub fn slice(&self, axis: usize, index: usize) -> Self {
        let res_shape = self.reduce_shape(axis);
        let res_values = self.slice_vector(axis, index);
        Self::new(res_values, res_shape)
    }

    // fn reduce(&self, axis: usize, reduce_f: &dyn Fn((&mut T, &T)) -> T) -> Self {
    //     let res_shape = self.reduce_shape(axis);
    //     let mut res_values = self.slice_vector(axis, 0);
    //     for idx in 1..(self.shape[axis]) {
    //         let add_values = self.slice_vector(axis, idx);
    //         res_values.iter_mut().zip(add_values.iter()).map(reduce_f).count();
    //     }
    //     Self::new(res_values, res_shape)
    // }

    pub fn sum(&self, axis: usize) -> Self {
        let n_prefix = count_elements(&self.shape[0..axis]);
        let n_axis_suffix = count_elements(&self.shape[axis..]);
        let n_suffix = count_elements(&self.shape[(axis + 1)..]);
        let array = self.values.borrow();

        let res_shape = self.reduce_shape(axis);
        let mut res_values = self.slice_vector(axis, 0);
        for prefix_idx in 0..n_prefix {
            for index in 1..self.shape[axis] {
                let src_start_idx = (prefix_idx * n_axis_suffix) + (index * n_suffix);
                let src_end_idx = src_start_idx + n_suffix;
                let res_start_idx = prefix_idx * n_suffix;
                let res_end_idx = res_start_idx + n_suffix;
                inpl_add(
                    &mut res_values[res_start_idx..res_end_idx],
                    &array[src_start_idx..src_end_idx],
                );
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

    #[test]
    fn sum() {
        let arr = Array::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);

        let arr0 = arr.sum(0);
        let target0: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];
        compare_vecs(&target0, &arr0.values.borrow());

        let arr1 = arr.sum(1);
        let target1: Vec<f32> = vec![4.0, 6.0, 12.0, 14.0];
        compare_vecs(&target1, &arr1.values.borrow());

        let arr2 = arr.sum(2);
        let target2: Vec<f32> = vec![3.0, 7.0, 11.0, 15.0];
        compare_vecs(&target2, &arr2.values.borrow());
    }
}
