use neurlang::neurlang::{MemoryLayout, Shape};

struct TensorIterator<const N: usize> {
    shape: Shape,
    memory_layout: MemoryLayout,
    return_index: [usize; N],
    count: usize,
    max_count: usize,
}

impl<const N: usize> TensorIterator<N> {
    fn new(shape: Shape, memory_layout: MemoryLayout) -> TensorIterator<N> {
        let nelem = shape.nelem();
        TensorIterator {
            shape: shape,
            memory_layout: memory_layout,
            return_index: [0; N],
            count: 0,
            max_count: nelem,
        }
    }
}

impl<const N: usize> Iterator for TensorIterator<N> {
    type Item = [usize; N];
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max_count - 1{
            let res = self.return_index.clone();
            
            match self.memory_layout {
                MemoryLayout::ColumnMajor => {
                    let mut idx = 0;
                    if self.return_index[idx] < self.shape.dimensions[idx] {
                        self.return_index[idx] += 1;
                    }
                    while self.return_index[idx] == self.shape.dimensions[idx] {
                        self.return_index[idx] = 0;
                        self.return_index[idx + 1] += 1;
                        idx += 1;
                    }
                },
                MemoryLayout::RowMajor => {
                    let mut idx = N - 1;
                    if self.return_index[idx] < self.shape.dimensions[idx] {
                        self.return_index[idx] += 1;
                    }
                    while self.return_index[idx] == self.shape.dimensions[idx] {
                        self.return_index[idx] = 0;
                        self.return_index[idx - 1] += 1;
                        idx -= 1;
                    }
                },
            };
            
            self.count += 1;
            Some(res)
        } else if self.count == self.max_count - 1 {
            self.count += 1;
            Some(self.return_index.clone())
        } else {
            None
        }
    }
}

fn main() {
    let shape = Shape::new(vec![2, 2, 2]);
    let layout = MemoryLayout::RowMajor;
    let tensor_iter: TensorIterator<3> = TensorIterator::new(shape, layout);
    let indices = tensor_iter.into_iter().collect::<Vec<_>>();
    println!("{indices:?}");
}
