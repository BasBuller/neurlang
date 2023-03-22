use neurlang::neurlang::Shape;

fn main() {
    let axes_size = [4, 2, 1];
    println!("{:?}", Shape::linear_index_to_array_index(0, &axes_size));
    println!("{:?}", Shape::linear_index_to_array_index(1, &axes_size));
    println!("{:?}", Shape::linear_index_to_array_index(2, &axes_size));
    println!("{:?}", Shape::linear_index_to_array_index(3, &axes_size));
}
