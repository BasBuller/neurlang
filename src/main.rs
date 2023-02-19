#![allow(dead_code)]

use std::rc::Rc;
use ndarray::{prelude::*, IxDynImpl};

type ReduceAxis = usize;

#[derive(Debug, Clone, Copy)]
enum ReduceOp {
    Sum,
    Max
}

#[derive(Debug)]
enum AST<T: ExecuteAST> {
    // Leaf
    Value {
        value: T,
    },

    // Unary
    Negate {
        value: Rc<AST<T>>,
    },
    Exponential {
        value: Rc<AST<T>>,
    },
    Log {
        value: Rc<AST<T>>,
    },

    // Binary
    Add {
        left_value: Rc<AST<T>>,
        right_value: Rc<AST<T>>,
    },

    // Reduce
    Reduce {
        value: Rc<AST<T>>,
        dim: ReduceAxis,
        op: ReduceOp,
    },
}

impl<'a, T: ExecuteAST> AST<T> {
    // Unary
    fn negate(self: Rc<Self>) -> Rc<AST<T>> {
        Rc::new(AST::Negate { value: self })
    }
    fn exp(self: Rc<Self>) -> Rc<AST<T>> {
        Rc::new(AST::Exponential { value: self })
    }
    fn log(self: Rc<Self>) -> Rc<AST<T>> {
        Rc::new(AST::Log { value: self })
    }

    // Binary
    fn add(self: Rc<Self>, value: Rc<AST<T>>) -> Rc<AST<T>> {
        Rc::new(AST::Add {
            left_value: self,
            right_value: value,
        })
    }
    fn subtract(self: Rc<Self>, value: Rc<AST<T>>) -> Rc<AST<T>> {
        Rc::new(AST::Add {
            left_value: self,
            right_value: value.negate(),
        })
    }

    // Reduce
    fn reduce(self: Rc<Self>, dim: ReduceAxis, op: ReduceOp) -> Rc<AST<T>> {
        Rc::new(AST::Reduce {
            value: self,
            dim: dim,
            op: op,
        })
    }

    // Utils
    fn new(value: T) -> Rc<AST<T>> {
        Rc::new(AST::Value { value })
    }
    fn execute(&self) -> T {
        match self {
            AST::Value { value } => value.value_v(),
            AST::Negate { value } => value.execute().negate_v(),
            AST::Exponential { value } => value.execute().exp_v(),
            AST::Log { value } => value.execute().log_v(),
            AST::Add {
                left_value,
                right_value,
            } => left_value.execute().add_v(right_value.execute()),
            AST::Reduce { value, dim, op } => value.execute().reduce(*dim, *op),
        }
    }
}

trait ExecuteAST {
    // Leaf
    fn value_v(&self) -> Self;

    // Unary
    fn negate_v(&self) -> Self;
    fn exp_v(&self) -> Self;
    fn log_v(&self) -> Self;

    // Binary
    fn add_v(&self, right_value: Self) -> Self;

    // Reduce
    fn reduce(&self, dim: ReduceAxis, op: ReduceOp) -> Self;
}

type Value = f64;
impl ExecuteAST for Value {
    fn value_v(&self) -> Self {
        *self
    }
    fn negate_v(&self) -> Self {
        -*self
    }
    fn exp_v(&self) -> Self {
        self.exp()
    }
    fn log_v(&self) -> Self {
        self.ln()
    }
    fn add_v(&self, right_value: Self) -> Self {
        *self + right_value
    }
    fn reduce(&self, _dim: ReduceAxis, _op: ReduceOp) -> Self {
        *self
    }
}

impl ExecuteAST for Array<f32, Dim<IxDynImpl>> {
    fn value_v(&self) -> Self {
        self.clone()
    }    
    fn negate_v(&self) -> Self {
        -self
    }
    fn exp_v(&self) -> Self {
        self.mapv(f32::exp)
    }
    fn log_v(&self) -> Self {
        self.mapv(f32::ln)
    }
    fn add_v(&self, right_value: Self) -> Self {
        self + right_value
    }
    fn reduce(&self, axis: ReduceAxis, op: ReduceOp) -> Self {
        match op {
            ReduceOp::Sum => self.sum_axis(Axis(axis)),
            ReduceOp::Max => self.fold_axis(Axis(axis), f32::MIN, |a, b| a.max(*b)),
        }
    }
}

fn main() {
    let ast = AST::new(5.0).subtract(AST::new(2.0)).add(AST::new(10.0));

    println!("{:?}", ast);
    println!("{:?}", ast.execute());
    
    let test = array![1.0, 2.0, 3.0];
    println!("{:?}", test.fold_axis(Axis(0), f32::MIN, |a, b| a.max(*b)));
}
