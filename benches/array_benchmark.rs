use criterion::{criterion_group, criterion_main, Criterion};
use neurlang::*;

fn unary_benchmark(c: &mut Criterion) {
    let shape: Shape = vec![100, 150];
    let array = rand_f32(shape.clone());
    let nelem = shape.iter().fold(1, |res, &val| res * val);
    let mut vec = (0..nelem).map(|val| val as f32 / 100.0).collect::<Vec<_>>();

    c.bench_function("negate array", |b| b.iter(|| array.negate()));
    c.bench_function("negate array inplace", |b| b.iter(|| array.inpl_negate()));
    c.bench_function("negate raw vector", |b| {
        b.iter(|| vec.iter().map(|&val| -val).collect::<Vec<_>>())
    });
    c.bench_function("negate raw vector inplace", |b| {
        b.iter(|| vec.iter_mut().map(|val| *val = -(*val)).count())
    });

    c.bench_function("exp array", |b| b.iter(|| array.exp()));
    c.bench_function("exp array inplace", |b| b.iter(|| array.inpl_exp()));
    c.bench_function("exp raw vector", |b| {
        b.iter(|| vec.iter().map(|val| val.exp()).collect::<Vec<_>>())
    });
    c.bench_function("exp raw vector inplace", |b| {
        b.iter(|| vec.iter_mut().map(|val| *val = val.exp()).count())
    });

    c.bench_function("ln array", |b| b.iter(|| array.ln()));
    c.bench_function("ln array inplace", |b| b.iter(|| array.inpl_ln()));
    c.bench_function("ln raw vector", |b| {
        b.iter(|| vec.iter().map(|val| val.ln()).collect::<Vec<_>>())
    });
    c.bench_function("ln raw vector inplace", |b| {
        b.iter(|| vec.iter_mut().map(|val| *val = val.ln()).count())
    });
}

fn binary_benchmark(c: &mut Criterion) {
    let shape = vec![100, 100];
    let nelem = shape.iter().fold(1, |res, &val| res * val);
    let array1 = rand_f32(shape.clone());
    let array2 = rand_f32(shape.clone());
    let vec1 = (0..nelem).map(|val| val as f32 / 100.0).collect::<Vec<f32>>();
    let vec2 = (0..nelem).map(|val| val as f32 / 100.0).collect::<Vec<f32>>();

    c.bench_function("add arrays", |b| b.iter(|| array1.add(&array2)));
    c.bench_function("add arrays manual", |b| b.iter(|| {
        let vs1 = array1.values.borrow();
        let vs2 = array2.values.borrow();
        let _res = vs1.iter().zip(vs2.iter()).map(|(&lval, &rval)| lval + rval).collect::<Vec<_>>();
    }));
    c.bench_function("add vectors", |b| b.iter(|| {
        let _res = vec1.iter().zip(vec2.iter()).map(|(&v1, &v2)| v1 + v2).collect::<Vec<_>>();
    }));
}

criterion_group!(benches, unary_benchmark, binary_benchmark);
criterion_main!(benches);
