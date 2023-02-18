#[derive(Debug)]
enum AST<'a, T: ExecuteAST> {
    Value {
        value: T,
    },
    Negate {
        value: &'a AST<'a, T>,
    },
    Add {
        left_value: &'a AST<'a, T>,
        right_value: &'a AST<'a, T>,
    },
}

impl<'a, T: ExecuteAST> AST<'a, T> {
    fn new(value: T) -> Box<AST<'a, T>> {
        Box::new(AST::Value { value })
    }

    fn negate(&'a self) -> Box<AST<'a, T>> {
        Box::new(AST::Negate { value: self })
    }

    fn add(&'a self, value: &'a AST<'a, T>) -> Box<AST<'a, T>> {
        Box::new(AST::Add {
            left_value: self,
            right_value: value,
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
    let temp = &AST::new(2.0);
    let ast = &AST::new(5.0);
    let ast = ast.negate();
    let ast = ast.add(temp);

    println!("{:?}", ast);
    println!("{:?}", ast.execute());
}
