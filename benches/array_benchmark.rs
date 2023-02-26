use criterion::{criterion_group, criterion_main, Criterion};
use neurlang::*;

fn negate_benchmark(c: &mut Criterion) {
    let shape: Shape = vec![10, 20, 30];
    let array = rand_f32(shape);
    
    c.bench_function("negate array", |b| b.iter(|| array.negate()));
}

fn add_benchmark(c: &mut Criterion) {
    let array1 = rand_f32(vec![10, 20, 30]);
    let array2 = rand_f32(vec![10, 20, 30]);
    
    c.bench_function("add arrays", |b| b.iter(|| array1.add(&array2)));
}

criterion_group!(benches, negate_benchmark, add_benchmark);
criterion_main!(benches);
