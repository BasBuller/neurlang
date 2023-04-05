use neurlang::array::*;
use neurlang::neurlang::*;

fn main() {
    let arr0 = ASTNode::new(
        Array::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        Shape::new(vec![2, 3]),
    );
    let arr1 = ASTNode::new(
        Array::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        Shape::new(vec![2, 3]),
    );
    let res = arr0
        .binary(&arr1, BinaryOp::Add)
        .unary(UnaryOp::Exponential)
        .reduce(0, ReduceOp::Sum);
    println!("{:?}", res);
}
