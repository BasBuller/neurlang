#![allow(dead_code)]

use std::rc::Rc;
use ndarray::{prelude::*, IxDynImpl};

type ReduceAxis = usize;

#[derive(Debug, Clone, Copy)]
enum ReduceOp {
    Sum,
    Max
}

type Shape = Vec<usize>;

#[derive(Debug)]
enum ASTOp<T: ExecuteAST> {
    // Leaf
    Value {
        value: T,
    },

    // Unary
    Negate {
        value: Rc<ASTOp<T>>,
    },
    Exponential {
        value: Rc<ASTOp<T>>,
    },
    Log {
        value: Rc<ASTOp<T>>,
    },

    // Binary
    Add {
        left_value: Rc<ASTOp<T>>,
        right_value: Rc<ASTOp<T>>,
    },

    // Reduce
    Reduce {
        value: Rc<ASTOp<T>>,
        dim: ReduceAxis,
        op: ReduceOp,
    },
}

impl<'a, T: ExecuteAST> ASTOp<T> {
    // Unary
    fn negate(self: Rc<Self>) -> Rc<ASTOp<T>> {
        Rc::new(ASTOp::Negate { value: self })
    }
    fn exp(self: Rc<Self>) -> Rc<ASTOp<T>> {
        Rc::new(ASTOp::Exponential { value: self })
    }
    fn log(self: Rc<Self>) -> Rc<ASTOp<T>> {
        Rc::new(ASTOp::Log { value: self })
    }

    // Binary
    fn add(self: Rc<Self>, right_value: Rc<ASTOp<T>>) -> Rc<ASTOp<T>> {
        Rc::new(ASTOp::Add {
            left_value: self,
            right_value: right_value,
        })
    }
    fn subtract(self: Rc<Self>, right_value: Rc<ASTOp<T>>) -> Rc<ASTOp<T>> {
        Rc::new(ASTOp::Add {
            left_value: self,
            right_value: right_value.negate(),
        })
    }

    // Reduce
    fn reduce(self: Rc<Self>, dim: ReduceAxis, op: ReduceOp) -> Rc<ASTOp<T>> {
        Rc::new(ASTOp::Reduce {
            value: self,
            dim: dim,
            op: op,
        })
    }

    // Utils
    fn new(value: T) -> Rc<ASTOp<T>> {
        Rc::new(ASTOp::Value { value })
    }
    fn execute(&self) -> T {
        match self {
            ASTOp::Value { value } => value.value_v(),
            ASTOp::Negate { value } => value.execute().negate_v(),
            ASTOp::Exponential { value } => value.execute().exp_v(),
            ASTOp::Log { value } => value.execute().log_v(),
            ASTOp::Add {
                left_value,
                right_value,
            } => left_value.execute().add_v(right_value.execute()),
            ASTOp::Reduce { value, dim, op } => value.execute().reduce(*dim, *op),
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
    let ast = ASTOp::new(5.0).subtract(ASTOp::new(2.0)).add(ASTOp::new(10.0));

    println!("{:?}", ast);
    println!("{:?}", ast.execute());
    
    let test = array![1.0, 2.0, 3.0];
    println!("{:?}", test.fold_axis(Axis(0), f32::MIN, |a, b| a.max(*b)));
}
