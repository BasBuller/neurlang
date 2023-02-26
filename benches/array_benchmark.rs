use criterion::{criterion_group, criterion_main, Criterion};
use neurlang::*;

fn setup(shape: Shape) -> (Array<f32>, Vec<f32>) {
    let nelem = shape.iter().fold(1, |res, &val| res * val);
    let array = rand_f32(shape.clone());
    let vec = (0..nelem).map(|val| val as f32 / 100.0).collect::<Vec<_>>();

    (array, vec)
}

fn negate_benchmark(c: &mut Criterion) {
    let (array, mut vec) = setup(vec![100, 150]);

    let mut group = c.benchmark_group("Negate");
    group.bench_function("Array, new object", |b| b.iter(|| array.negate()));
    group.bench_function("Vector, new object", |b| {
        b.iter(|| vec.iter().map(|&val| -val).collect::<Vec<_>>())
    });

    group.bench_function("Array, in place", |b| b.iter(|| array.inpl_negate()));
    group.bench_function("Vector, in place", |b| {
        b.iter(|| vec.iter_mut().map(|val| *val = -(*val)).count())
    });
    group.finish();
}

fn exp_benchmark(c: &mut Criterion) {
    let (array, mut vec) = setup(vec![100, 150]);

    let mut group = c.benchmark_group("Exponate");
    group.bench_function("Array, new object", |b| b.iter(|| array.exp()));
    group.bench_function("Vector, new object", |b| {
        b.iter(|| vec.iter().map(|val| val.exp()).collect::<Vec<_>>())
    });

    group.bench_function("Array, in place", |b| b.iter(|| array.inpl_exp()));
    group.bench_function("Vector, in place", |b| {
        b.iter(|| vec.iter_mut().map(|val| *val = val.exp()).count())
    });
    group.finish();
}

fn ln_benchmark(c: &mut Criterion) {
    let (array, mut vec) = setup(vec![100, 150]);

    let mut group = c.benchmark_group("Natural logarithm");
    group.bench_function("Array, new object", |b| b.iter(|| array.ln()));
    group.bench_function("Vector, new object", |b| {
        b.iter(|| vec.iter().map(|val| val.ln()).collect::<Vec<_>>())
    });

    group.bench_function("Array, in place", |b| b.iter(|| array.inpl_ln()));
    group.bench_function("Vector, in place", |b| {
        b.iter(|| vec.iter_mut().map(|val| *val = val.ln()).count())
    });
    group.finish();
}

fn binary_benchmark(c: &mut Criterion) {
    let (array0, vec0) = setup(vec![100, 150]);
    let (array1, vec1) = setup(vec![100, 150]);

    let mut group = c.benchmark_group("Add");
    group.bench_function("Array", |b| b.iter(|| array0.add(&array1)));
    group.bench_function("Vector", |b| {
        b.iter(|| {
            let _ = vec0
                .iter()
                .zip(vec1.iter())
                .map(|(&v1, &v2)| v1 + v2)
                .collect::<Vec<_>>();
        })
    });
}

criterion_group!(
    benches,
    negate_benchmark,
    exp_benchmark,
    ln_benchmark,
    binary_benchmark
);
criterion_main!(benches);
