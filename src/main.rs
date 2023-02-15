use num::pow;

#[derive(Debug)]
enum AST {
    Value {
        value: f64,
    },
    Negate {
        value: Box<AST>,
    },
    //Exp { value: Box<AST> },
    Add {
        left_value: Box<AST>,
        right_value: Box<AST>,
    },
    Multiply {
        left_value: Box<AST>,
        right_value: Box<AST>,
    },
}

impl AST {
    fn new(value: f64) -> Box<AST> {
        Box::new(AST::Value { value })
    }

    fn negate(value: Box<AST>) -> Box<AST> {
        Box::new(AST::Negate { value })
    }

    //fn exp(value: Box<AST>) -> Box<AST> {
    //    Box::new(AST::Exp { value })
    //}

    fn add(left_value: Box<AST>, right_value: Box<AST>) -> Box<AST> {
        Box::new(AST::Add {
            left_value,
            right_value,
        })
    }

    fn subtract(left_value: Box<AST>, right_value: Box<AST>) -> Box<AST> {
        AST::add(left_value, AST::negate(right_value))
    }

    fn multiply(left_value: Box<AST>, right_value: Box<AST>) -> Box<AST> {
        Box::new(AST::Multiply {
            left_value,
            right_value,
        })
    }

    fn execute(&self) -> f64 {
        match self {
            AST::Value { value } => *value,
            AST::Negate { value } => -value.execute(),
            AST::Add {
                left_value,
                right_value,
            } => left_value.execute() + right_value.execute(),
            AST::Multiply {
                left_value,
                right_value,
            } => left_value.execute() * right_value.execute(),
        }
    }
}

fn main() {
    let ast = AST::multiply(
        AST::add(AST::negate(AST::new(5.0)), AST::new(10.0)),
        AST::new(2.0),
    );

    println!("{:?}", ast);
    println!("{:?}", ast.execute());
}
