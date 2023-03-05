// use neurlang::*;

pub fn square<T>(values: &Vec<T>) -> Vec<T>
where
    T: std::ops::Mul + std::ops::Mul<Output = T> + Copy,
{
    values
        .iter()
        .map(|&value| value * value)
        .collect::<Vec<_>>()
}

pub fn square_f32(values: &Vec<f32>) -> Vec<f32> {
    values
        .iter()
        .map(|&value| value * value)
        .collect::<Vec<_>>()
}

fn main() {
    let vec0 = vec![0.0, 1.0, 2.0, 3.0];
    let res = square::<f32>(&vec0);
}
