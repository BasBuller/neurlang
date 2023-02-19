use std::rc::Rc;

#[derive(Debug)]
enum AST<T: ExecuteAST> {
    Value {
        value: T,
    },
    Negate {
        value: Rc<AST<T>>,
    },
    Add {
        left_value: Rc<AST<T>>,
        right_value: Rc<AST<T>>,
    },
}

impl<'a, T: ExecuteAST> AST<T> {
    fn new(value: T) -> Rc<AST<T>> {
        Rc::new(AST::Value { value })
    }

    fn negate(self: Rc<Self>) -> Rc<AST<T>> {
        Rc::new(AST::Negate { value: self })
    }

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

    fn execute(&self) -> T {
        match self {
            AST::Value { value } => value.value(),
            AST::Negate { value } => value.execute().negate(),
            AST::Add {
                left_value,
                right_value,
            } => left_value.execute().add(right_value.execute()),
        }
    }
}

trait ExecuteAST {
    fn value(&self) -> Self;
    fn negate(&self) -> Self;
    fn add(&self, right_value: Self) -> Self;
}

type Value = f64;
impl ExecuteAST for Value {
    fn value(&self) -> Self {
        *self
    }
    fn negate(&self) -> Self {
        -*self
    }
    fn add(&self, right_value: Self) -> Self {
        *self + right_value
    }
}

fn main() {
    let ast = AST::new(2.0).negate().add(AST::new(5.0));

    println!("{:?}", ast);
    println!("{:?}", ast.execute());
}
