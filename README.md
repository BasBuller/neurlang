# Neurlang - Have fun exploring how linear algebra libraries are implemented while building something cool
Much of the research in accelerated linear algebra and deep learning frameworks is focussed on enabling ever larger models on increasingly more hardware of increasing complexity. Whereas the models coming out of this process are amazing, it is far from accessible or practical if you do not have a huge pile of cash to burn, read pretty much everyone.

## Features (as far as they are implemented)
- Computational graph is constructed ahead of execution time. This allows for shape checking without having to run intensive compute. It also sets us up for implementing graph optimisations. Inspired by how Jax & Tensorflow operate, and as of recently PyTorch 2.0.
- Only rely on a small number of base operations, inspired by Tinygrad. Currently very unoptimised Rust implementations are available.

## Very basic example
```rust
use neurlang::array::Array;
use neurlang::neurlang::{ASTNode, Shape, BinaryOp};

fn main() {
    let weight_matrix = ASTNode::new(
        Array::new((1..16).map(|x| x as f32).collect::<Vec<_>>()),
        Shape::new(vec![3, 5]),
    );
    let bias_vec = ASTNode::new(
        Array::new(vec![1.0, 1.0, 1.0]),
        Shape::new(vec![3]),
    );
    let inputs = ASTNode::new(
        Array::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        Shape::new(vec![5]),
    );

    let results_ast = weight_matrix
        .tensordot(&inputs, 1)
        .binary(&bias_vec, BinaryOp::Add);
    let results = results_ast.execute();
}
```

## A non-exhaustive and very rough list of reasons why this project exists:
- Most importantly, have fun and learn about the wildly interesting landscape of computation focussed on ML applications, both for large data centers and on device!
- Trace computational graph to allow for optimizations
    - Maybe, one day, make graph tracing JIT, allowing for lazy optimizations yet a dynamic experience
    - See if I can implement something similar to the broadcasting operator like in Julia and Matlab, ie vector .+ 5
- CPU first, followed by mobile platforms
    - Little interest in enabling CUDA for now, given the bazillion alternatives
- Focus on experimentation with new compute and ML paradigms, not being another generic DL framework. If that's what you're looking for, have a look at PyTorch, Jax, Tinygrad, dfdx...
    - Does not mean I am not taking inspiration from those, think of a JIT compiler or the bare minimal number of ops like using in Tinygrad.
- Some paradigms I would like to explore one day:
    - Very sparse neural networks
    - Asynchronous and distributed training and inference
    - Run and train on widely available comodity hardware
    - Spiking neural networks
    - Lazy computational graph optimization
    - Blur the line between high performance numerical computing and general purposing computing
- Desire and inspiration to experiment with distributed training on heterogeneous chips comes from the following systems:
    - [Pathways](https://arxiv.org/abs/2203.12533), Google their propietary (and experimental?) distributed and asynchronous trainig framework. Find this reall exciting and hope to do some work towards this.
    - [Ray](https://docs.ray.io/en/latest/train/train.html), but instead build everything in one language instead of creating a runtime on top of existing DL frameworks
    - [LLama.cpp](https://github.com/ggerganov/llama.cpp), seeing large models move to on-device is a really big thing. This is the first step towards everyone having access to their own personal AIs, running on commodity hardware, adjusting to your personal needs, and working with your private data without sending it off to a third party.
- Be cool an use Rust. Jokes aside, it's in interesting language that lends itself well to a high-performance and distributed application like this
