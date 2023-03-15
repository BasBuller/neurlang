## How to enable reduce operations
The following is necessary to properly perform a reduction:
- Stride - The number of elements between two elements in the reduction dimension. This is the number of elements to go from item n in the first dim to item n+1, also **product of the dimension sizes except for the first dimension**.
- Offset - The index of the first item in the target dimension within the first row, also **product of the dimension sizes for dimensions that come after the reduce dimension**.

## How to implement matrix vector multiplication
Most effective way to think about a Matrix is as a linear function:
    $$f: R^{n} \rightarrow R^{m} \newline Matrix(f) = R^{m} \times R^{n}$$
    
When performing matrix multiplication you ideally want to iterate over the **rows of the left matrix** and iterate over the **columns of the right matrix**. This means the left preferably is *row major* and the right matrix is *column major*.

## Ideas for implementations
- Can I express the ideal array configurations for an operation as **constraints on the arrays**? Like using a trait bound on the left and right arrays for matrix multiplication that they have row and column major memory layouts respectively? Or maybe it's just as attribute of an object and my program is free to optimize over different options for the attribute, ie an enum. 
    - The Same way of adding constraints can be done for binary ops like addition, where the arrays need the same memory layouts.
    - By defining these traits at my AST level I can implement them for each different type of accelerator, ie CPU, GPU1, GPU2, my own custom chip, ...
    - These relations/constraints can be very well captured in a domain-specific mathematical language, this is exactly what category theory excels at. Would be very cool to have a form of mathematical reasoning enabling me to highly optimize my ASTs. I probably do need an estimate of the cost of specific attribute combinations in order to turn graph refinement into a proper optimization problem.
    
## To Do's
- [x] Update Shape implementation used in Array
- [x] Implement very first version of expand
    - [x] Improve expand performance by preventing a copy, did so by enabling a Rc around the backing array
- [ ] Implement permutation/reshape
- [ ] Implement matmul in pure Rust
- [ ] Maybe I need to implement the tensorproduct like in Numpy? That enables essentially all types of tensor operations
    - [ ] If done properly this can also be rather speedy, because I can write logic to reason about how to optimize the iteration
- [ ] Enable BLAS matmul
- [ ] Greatly improve tensor indexing code quality

### When implement column major ordering, if at all?
- [ ] Enable slicing in column major arrays by taking $array_len - row_major_indexer$
- [ ] Implement conversion between row and column major storage formats
    - [x] Implement indexing and then simply create an indices iterator
