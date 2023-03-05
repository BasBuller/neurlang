use criterion::{criterion_group, criterion_main, Criterion};
use neurlang::*;

fn setup(shape: Shape) -> (Array<f32>, Vec<f32>) {
    let nelem = shape.iter().fold(1, |res, &val| res * val);
    let array = rand_f32(shape.clone());
    let vec = (0..nelem).map(|val| val as f32 / 100.0).collect::<Vec<_>>();

    (array, vec)
}

fn negate_benchmark(c: &mut Criterion) {
    let shape = vec![512, 1024];
    let nelem = shape.iter().fold(1, |res, &val| res * val);
    let (array, mut vec) = setup(shape);

    let mut group = c.benchmark_group("Negate new object");
    group.bench_function("Array", |b| b.iter(|| array.negate()));
    group.bench_function("Vector", |b| b.iter(|| negate(&vec)));
    group.bench_function("Slice", |b| b.iter(|| negate(&vec[0..nelem])));
    group.finish();

    let mut group = c.benchmark_group("Negate in place");
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

fn reduce_benchmark(c: &mut Criterion) {
    let (array0, _) = setup(vec![100, 200, 150]);

    let mut group = c.benchmark_group("Sum");
    group.bench_function("Array", |b| b.iter(|| array0.sum(1)));
}

criterion_group!(
    benches,
    negate_benchmark,
    exp_benchmark,
    ln_benchmark,
    add_benchmark,
    reduce_benchmark,
);
criterion_main!(benches);
