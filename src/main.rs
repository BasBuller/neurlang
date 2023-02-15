type Value = f64;

#[derive(Debug)]
enum AST<'a> {
    Value {
        value: Value,
    },
    Negate {
        value: &'a AST<'a>,
    },
    Add {
        left_value: &'a AST<'a>,
        right_value: &'a AST<'a>,
    },
}

impl<'a> AST<'a> {
    fn new(value: Value) -> Box<AST<'a>> {
        Box::new(AST::Value { value })
    }
    
    fn negate(&'a self) -> Box<AST<'a>> {
        Box::new(AST::Negate { value: self })
    }
    
    fn add(&'a self, value: &'a AST) -> Box<AST<'a>> {
        Box::new(AST::Add { left_value: self, right_value: value })
    }
    
    fn execute_cpu(&self) -> Value {
        match self {
            AST::Value { value } => *value,
            AST::Negate { value } => -value.execute_cpu(),
            AST::Add { left_value, right_value } => left_value.execute_cpu() + right_value.execute_cpu(),
        }
    }
}

fn main() {
    let temp = &AST::new(2.0);
    let ast = &AST::new(5.0);
    let ast = ast.negate();
    let ast = ast.add(temp);

    println!("{:?}", ast);
    println!("{:?}", ast.execute_cpu());
}
