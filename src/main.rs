type Value = f64;
impl Lazy for Value {
    fn execute(&self) -> Value {
        *self
    }
}


trait Lazy {
    fn execute(&self) -> Value;
}
trait UnaryOp<T: Lazy> {
    fn new(value: Box<T>) -> Box<Self>;
}
trait BinaryOp<T: Lazy, U: Lazy> {
    fn new(left_value: Box<T>, right_value: Box<U>) -> Box<Self>;
}


#[derive(Debug)]
struct Negate<T: Lazy> {
    value: Box<T>,
}
impl<'a, T: Lazy> UnaryOp<T> for Negate<T> {
    fn new(value: Box<T>) -> Box<Negate<T>> {
        Box::new(Negate{value: value})
    }
}
impl<T: Lazy> Lazy for Negate<T> {
    fn execute(&self) -> Value {
        self.value.execute() * -1.0
    }
}


#[derive(Debug)]
struct Add<T: Lazy, U: Lazy> {
    left_value: Box<T>,
    right_value: Box<U>,
}
impl<'a, T: Lazy, U: Lazy> BinaryOp<T, U> for Add<T, U> {
    fn new(left_value: Box<T>, right_value: Box<U>) -> Box<Add<T, U>> {
        Box::new(Add{left_value: left_value, right_value: right_value})
    }
}
impl<'a, T: Lazy, U: Lazy> Lazy for Add<T, U> {
    fn execute(&self) -> Value {
        self.left_value.execute() + self.right_value.execute()
    }
}

fn subtract<T: Lazy, U: Lazy>(left_value: Box<T>, right_value: Box<U>) -> Box<Add<T, Negate<U>>> {
    let neg_right_value = Negate::new(right_value);
    Add::new(left_value, neg_right_value)
}


fn main() {
    let ast0 = Negate::new(Box::new(1.0));
    let ast1 = Add::new(ast0, Box::new(5.0));
    let ast = subtract(ast1, Box::new(10.0));
    println!("{:?}", ast);
    println!("{:?}", ast.execute());
}
