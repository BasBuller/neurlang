use neurlang::array::*;

use criterion::{criterion_group, criterion_main, Criterion};
use num::Float;

// Some reference implementations
pub fn negate_arr_raw<const N: usize>(values: &[f32; N], target: &mut [f32; N]) {
    values
        .iter()
        .zip(target.iter_mut())
        .for_each(|(&lval, res)| *res = -lval);
}
pub fn negate<T: Float>(values: &[T]) -> Vec<T> {
    values.iter().map(|&value| -value).collect::<Vec<_>>()
}
pub fn inpl_negate_arr_raw<const N: usize>(values: &mut [f32; N]) {
    values.iter_mut().for_each(|value| *value = -(*value));
}
pub fn inpl_negate<T: Float>(values: &mut [T]) {
    values.iter_mut().for_each(|value| *value = -(*value));
}

pub fn exp<T: Float>(values: &[T]) -> Vec<T> {
    values.iter().map(|value| value.exp()).collect::<Vec<_>>()
}
pub fn inpl_exp<T: Float>(values: &mut [T]) {
    values.iter_mut().map(|value| *value = value.exp()).count();
}

pub fn ln<T: Float>(values: &[T]) -> Vec<T> {
    values.iter().map(|value| value.ln()).collect::<Vec<_>>()
}
pub fn inpl_ln<T: Float>(values: &mut [T]) {
    values.iter_mut().map(|value| *value = value.ln()).count();
}

pub fn binary_op<T, F>(lvalues: &[T], rvalues: &[T], binary_f: F) -> Vec<T>
where
    T: Float,
    F: Fn((&T, &T)) -> T,
{
    lvalues
        .iter()
        .zip(rvalues.iter())
        .map(binary_f)
        .collect::<Vec<_>>()
}

fn setup<const N: usize>(dimensions: [usize; N]) -> (Array<f32, N>, Vec<f32>) {
    let shape = Shape::new(dimensions);
    let array = rand_f32(shape.clone());
    let vec = (0..shape.nelem())
        .map(|val| val as f32 / 100.0)
        .collect::<Vec<_>>();

    (array, vec)
}

fn negate_benchmark(c: &mut Criterion) {
    let shape = [5000, 5000];
    let (array, mut vec) = setup(shape);

    let mut group = c.benchmark_group("Negate new object");
    group.bench_function("Array", |b| b.iter(|| array.negate()));
    group.bench_function("Vector", |b| b.iter(|| negate(&vec)));
    group.finish();

    let mut group = c.benchmark_group("Negate in place");
    group.bench_function("Array", |b| b.iter(|| array.inpl_negate()));
    group.bench_function("Vector", |b| b.iter(|| inpl_negate(&mut vec)));
    group.finish();
}

fn exp_benchmark(c: &mut Criterion) {
    let (array, mut vec) = setup([5000, 5000]);

    let mut group = c.benchmark_group("Exponate new object");
    group.bench_function("Array", |b| b.iter(|| array.exp()));
    group.bench_function("Vector", |b| b.iter(|| exp(&vec)));
    group.finish();

    let mut group = c.benchmark_group("Exponate in place");
    group.bench_function("Array", |b| b.iter(|| array.inpl_exp()));
    group.bench_function("Vector", |b| b.iter(|| inpl_exp(&mut vec)));
    group.finish();
}

fn ln_benchmark(c: &mut Criterion) {
    let (array, mut vec) = setup([5000, 5000]);

    let mut group = c.benchmark_group("Natural logarithm new object");
    group.bench_function("Array", |b| b.iter(|| array.ln()));
    group.bench_function("Vector", |b| b.iter(|| ln(&vec)));
    group.finish();

    let mut group = c.benchmark_group("Natural logarithm in place");
    group.bench_function("Array", |b| b.iter(|| array.inpl_ln()));
    group.bench_function("Vector", |b| b.iter(|| inpl_ln(&mut vec)));
    group.finish();
}

fn add_benchmark(c: &mut Criterion) {
    let (array0, vec0) = setup([5000, 5000]);
    let (array1, vec1) = setup([5000, 5000]);

    let mut group = c.benchmark_group("Add");
    group.bench_function("Array", |b| b.iter(|| array0.add(&array1)));
    group.bench_function("Vector", |b| {
        b.iter(|| binary_op(&vec0, &vec1, |(&lval, &rval)| lval + rval))
    });
}

fn multiply_benchmark(c: &mut Criterion) {
    let (array0, vec0) = setup([5000, 5000]);
    let (array1, vec1) = setup([5000, 5000]);

    let mut group = c.benchmark_group("Multiply");
    group.bench_function("Array", |b| b.iter(|| array0.multiply(&array1)));
    group.bench_function("Vector", |b| {
        b.iter(|| binary_op(&vec0, &vec1, |(&lval, &rval)| lval * rval))
    });
}

fn slice_benchmark(c: &mut Criterion) {
    let (array0, _) = setup([128, 512, 1024]);

    c.bench_function("Slice array", |b| b.iter(|| array0.slice(1, 0)));
}

fn reduce_benchmark(c: &mut Criterion) {
    let (array0, _) = setup([128, 512, 1024]);

    c.bench_function("Sum array dim", |b| b.iter(|| array0.reduce_sum(1)));
    c.bench_function("Max array dim", |b| b.iter(|| array0.reduce_max(1)));
}

fn squeeze_unsqueeze_benchmark(c: &mut Criterion) {
    let (array, _) = setup([1024, 2048, 1]);

    c.bench_function("Unsqueeze array", |b| b.iter(|| array.unsqueeze(3)));
    c.bench_function("Squeeze array", |b| b.iter(|| array.squeeze(2)));
}

fn permute_benchmark(c: &mut Criterion) {
    let (array, _) = setup([5000, 5000]);
    c.bench_function("Permute", |b| b.iter(|| array.permute([1, 0])));
}

fn pad_benchmark(c: &mut Criterion) {
    let (array, _) = setup([5000, 5000]);
    let padding = [PadAxis(2, 2, 0.0), PadAxis(2, 2, 0.0)];
    c.bench_function("Padding", |b| b.iter(|| array.pad(padding)));
}

fn python_compare(c: &mut Criterion) {
    let (_, vec0) = setup([5000, 5000]);
    let (_, vec1) = setup([5000, 5000]);

    c.bench_function("Python compare", |b| {
        b.iter(|| binary_op(&vec0, &vec1, |(&lval, &rval)| lval * rval + lval * rval))
    });
}

criterion_group!(
    benches,
    negate_benchmark,
    exp_benchmark,
    ln_benchmark,
    add_benchmark,
    multiply_benchmark,
    slice_benchmark,
    reduce_benchmark,
    squeeze_unsqueeze_benchmark,
    permute_benchmark,
    python_compare,
    pad_benchmark,
);
criterion_main!(benches);
