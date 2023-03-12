use neurlang::array::*;
use neurlang::neurlang::Shape;

use criterion::{criterion_group, criterion_main, Criterion};
use num::Float;

// Some reference implementations
pub fn negate_arr_raw<const N: usize>(values: &[f32; N], target: &mut [f32; N]) {
    values.iter().zip(target.iter_mut()).for_each(|(&lval, res)| *res = -lval);
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

pub fn add<T: Float>(lvalues: &[T], rvalues: &[T]) -> Vec<T> {
    lvalues
        .iter()
        .zip(rvalues.iter())
        .map(|(&lval, &rval)| lval + rval)
        .collect::<Vec<_>>()
}

fn setup(dimensions: Vec<usize>) -> (Array<f32>, Vec<f32>) {
    let shape = Shape::new(dimensions);
    let nelem = shape.nelem();
    let array = rand_f32(shape.clone());
    let vec = (0..nelem).map(|val| val as f32 / 100.0).collect::<Vec<_>>();

    (array, vec)
}

fn negate_benchmark(c: &mut Criterion) {
    let shape = vec![512, 1024];
    let nelem = shape.iter().fold(1, |res, &val| res * val);
    let (array, mut vec) = setup(shape);
    
    const N_ELEM: usize = 512 * 1024;
    let raw_arr = Box::new([1.0; N_ELEM]);
    let mut target_raw_arr = Box::new([1.0; N_ELEM]);

    let mut group = c.benchmark_group("Negate new object");
    group.bench_function("Raw Array", |b| b.iter(|| negate_arr_raw(&raw_arr, &mut target_raw_arr)));
    group.bench_function("Array", |b| b.iter(|| array.negate()));
    group.bench_function("Vector", |b| b.iter(|| negate(&vec)));
    group.bench_function("Slice", |b| b.iter(|| negate(&vec[0..nelem])));
    group.finish();

    let mut group = c.benchmark_group("Negate in place");
    group.bench_function("Raw Array", |b| b.iter(|| inpl_negate_arr_raw(&mut target_raw_arr)));
    group.bench_function("Array", |b| b.iter(|| array.inpl_negate()));
    group.bench_function("Vector", |b| b.iter(|| inpl_negate(&mut vec)));
    group.bench_function("Slice", |b| b.iter(|| inpl_negate(&mut vec[0..nelem])));
    group.finish();
}

fn exp_benchmark(c: &mut Criterion) {
    let (array, mut vec) = setup(vec![512, 1024]);

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
    let (array, mut vec) = setup(vec![512, 1024]);

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
    let (array0, vec0) = setup(vec![512, 1024]);
    let (array1, vec1) = setup(vec![512, 1024]);

    let mut group = c.benchmark_group("Add");
    group.bench_function("Array", |b| b.iter(|| array0.add(&array1)));
    group.bench_function("Vector", |b| b.iter(|| add(&vec0, &vec1)));
}

fn slice_benchmark(c: &mut Criterion) {
    let (array0, _) = setup(vec![128, 512, 1024]);

    c.bench_function("Slice array", |b| b.iter(|| array0.slice(1, 0)));
}

fn reduce_benchmark(c: &mut Criterion) {
    let (array0, _) = setup(vec![128, 512, 1024]);

    c.bench_function("Sum array dim", |b| b.iter(|| array0.reduce_sum(1)));
    c.bench_function("Max array dim", |b| b.iter(|| array0.reduce_max(1)));
}

criterion_group!(
    benches,
    negate_benchmark,
    exp_benchmark,
    ln_benchmark,
    add_benchmark,
    slice_benchmark,
    reduce_benchmark,
);
criterion_main!(benches);
