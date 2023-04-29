use neurlang::array::{Array, rand_f32};
use neurlang::neurlang::{ASTNode, Shape, BinaryOp};

fn main() {
    let weight_matrix = ASTNode::new(
        Array::new((1..16).map(|x| x as f32).collect::<Vec<_>>()),
        Shape::new(vec![3, 5]),
    );
    let bias_vec = ASTNode::new(Array::new(vec![1.0, 1.0, 1.0]), Shape::new(vec![3]));

    let input_shape = Shape::new(vec![5]);
    let inputs = ASTNode::new(rand_f32(&input_shape), input_shape);

    let results_ast = weight_matrix
        .tensordot(&inputs, 1)
        .binary(&bias_vec, BinaryOp::Add);
    let results = results_ast.execute();

    println!("{:?} -- {:?}", results, results_ast.shape);
}
