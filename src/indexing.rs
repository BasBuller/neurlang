use crate::neurlang::MemoryLayout;
use crate::array::{ArrayIndex, Shape};

pub fn make_slice(shape: &Shape, axis: usize, index: usize) -> Slice {
    Slice { shape, axis, index }
}

// //////////////////
// Slicing
// //////////////////
#[derive(Debug, Clone, Copy)]
pub struct Slice<'a> {
    shape: &'a Shape,
    axis: usize,
    index: usize,
}

pub struct SliceIterator {
    pub n_prefix: usize,
    pub n_axis_suffix: usize,
    pub n_suffix: usize,
    pub index: usize,
    prefix_idx: usize,
}

impl<'a> IntoIterator for Slice<'a> {
    type Item = (usize, usize);
    type IntoIter = SliceIterator;

    fn into_iter(self) -> Self::IntoIter {
        let axis = self.axis;
        let n_prefix = self.shape.dimensions[0..axis].iter().product();
        let n_axis_suffix = self.shape.dimensions[axis..].iter().product();
        let n_suffix = self.shape.dimensions[(axis + 1)..].iter().product();

        SliceIterator {
            n_prefix: n_prefix,
            prefix_idx: 0,
            n_axis_suffix: n_axis_suffix,
            n_suffix: n_suffix,
            index: self.index,
        }
    }
}

impl Iterator for SliceIterator {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.prefix_idx < self.n_prefix {
            let src_start_idx =
                (self.prefix_idx * self.n_axis_suffix) + (self.index * self.n_suffix);
            let src_end_idx = src_start_idx + self.n_suffix;
            self.prefix_idx += 1;
            Some((src_start_idx, src_end_idx))
        } else {
            None
        }
    }
}

// /////////////////////////
// Iterate over tensor indices
// /////////////////////////
pub struct TensorIterator {
    shape: Shape,
    memory_layout: MemoryLayout,
    return_index: Vec<usize>,
    count: usize,
    max_count: usize,
}

impl TensorIterator {
    pub fn new(shape: Shape, memory_layout: MemoryLayout) -> TensorIterator {
        let nelem = shape.nelem();
        let ndim = shape.len();
        TensorIterator {
            shape: shape,
            memory_layout: memory_layout,
            return_index: vec![0; ndim],
            count: 0,
            max_count: nelem,
        }
    }
}

impl Iterator for TensorIterator {
    type Item = ArrayIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max_count - 1 {
            let res = ArrayIndex::new(self.return_index.clone());

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
                }
                MemoryLayout::RowMajor => {
                    let mut idx = self.shape.len() - 1;
                    if self.return_index[idx] < self.shape.dimensions[idx] {
                        self.return_index[idx] += 1;
                    }
                    while self.return_index[idx] == self.shape.dimensions[idx] {
                        self.return_index[idx] = 0;
                        self.return_index[idx - 1] += 1;
                        idx -= 1;
                    }
                }
            };

            self.count += 1;
            Some(res)
        } else if self.count == self.max_count - 1 {
            self.count += 1;
            Some(ArrayIndex::new(self.return_index.clone()))
        } else {
            None
        }
    }
}

// /////////////////////////
// Linear and re-orderd iteration
// /////////////////////////
pub struct PermutedTensorIterator<const N: usize> {
    shape: [usize; N],
    ordered_index_offset: [usize; N],
    permuted_shape: [usize; N],
    permutation: [usize; N],
    return_index: [usize; N],
    count: usize,
    max_count: usize,
    block_size: usize,
    n_ordered_trailing_axes: usize,
}

impl<const N: usize> PermutedTensorIterator<N> {
    pub fn new(shape: Shape, permutation: [usize; N]) -> Self {
        let mut block_size = 1;
        let mut n_ordered_trailing_axes = 0;
        for (dim1, &dim2) in (0..N).rev().zip(permutation.iter().rev()) {
            if dim1 == dim2 {
                block_size *= shape.dimensions[dim1];
                n_ordered_trailing_axes += 1;
            } else {
                break;
            }
        }

        let max_count = shape.nelem();
        let shape_array: [usize; N] = shape.dimensions.try_into().unwrap();
        let mut permuted_shape = [0; N];
        for (permute_idx, &original_idx) in permutation.iter().enumerate() {
            permuted_shape[permute_idx] = shape_array[original_idx];
        }
        let mut ordered_index_offset = [1; N];
        for dim_idx in 0..N {
            ordered_index_offset[dim_idx] = shape_array[(dim_idx + 1)..]
                .iter()
                .fold(1, |res, dim_size| res * dim_size);
        }

        PermutedTensorIterator {
            shape: shape_array,
            ordered_index_offset: ordered_index_offset,
            permuted_shape: permuted_shape,
            permutation: permutation,
            return_index: [0; N],
            count: 0,
            max_count: max_count,
            block_size: block_size,
            n_ordered_trailing_axes: n_ordered_trailing_axes,
        }
    }

    fn make_ordered_index_array(&self, permuted_index: &[usize; N]) -> [usize; N] {
        let mut res = [0; N];
        for (&idx, &value) in self.permutation.iter().zip(permuted_index.iter()) {
            res[idx] = value;
        }
        return res;
    }

    fn permute_index_array(&self, ordered_index: &mut [usize; N]) -> [usize; N] {
        let mut res = [0; N];
        for (&idx, &value) in self.permutation.iter().zip(ordered_index.iter()) {
            res[idx] = value;
        }
        return res;
    }
}

impl<const N: usize> Iterator for PermutedTensorIterator<N> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let ordered_index = self.make_ordered_index_array(&self.return_index);
        let linear_starting_index = ordered_index
            .iter()
            .zip(self.ordered_index_offset.iter())
            .fold(0, |res, (&index, &size)| res + index * size);
        let res = (
            linear_starting_index,
            linear_starting_index + self.block_size,
        );

        if self.count < self.max_count - self.block_size {
            // Calculate index for the next iteration
            let mut idx = N - 1 - self.n_ordered_trailing_axes;
            if self.return_index[idx] < self.permuted_shape[idx] {
                self.return_index[idx] += 1;
            }
            while self.return_index[idx] == self.permuted_shape[idx] {
                self.return_index[idx] = 0;
                self.return_index[idx - 1] += 1;
                idx -= 1;
            }

            self.count += self.block_size;
            Some(res)
        } else if self.count == self.max_count - self.block_size {
            self.count += self.block_size;
            Some(res)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slicing_iterator() {
        let shape = Shape::new(vec![2, 2, 2]);
        let slice = make_slice(&shape, 0, 0);
        let target_indices = vec![(0, 4)];
        let slice_indices = slice.into_iter().collect::<Vec<_>>();
        assert_eq!(target_indices, slice_indices);

        let shape = Shape::new(vec![2, 2, 2]);
        let slice = make_slice(&shape, 0, 1);
        let target_indices = vec![(4, 8)];
        let slice_indices = slice.into_iter().collect::<Vec<_>>();
        assert_eq!(target_indices, slice_indices);

        let shape = Shape::new(vec![2, 2, 2]);
        let slice = make_slice(&shape, 1, 0);
        let target_indices = vec![(0, 2), (4, 6)];
        let slice_indices = slice.into_iter().collect::<Vec<_>>();
        assert_eq!(target_indices, slice_indices);

        let shape = Shape::new(vec![2, 2, 2]);
        let slice = make_slice(&shape, 1, 1);
        let target_indices = vec![(2, 4), (6, 8)];
        let slice_indices = slice.into_iter().collect::<Vec<_>>();
        assert_eq!(target_indices, slice_indices);

        let shape = Shape::new(vec![2, 2, 2]);
        let slice = make_slice(&shape, 2, 0);
        let target_indices = vec![(0, 1), (2, 3), (4, 5), (6, 7)];
        let slice_indices = slice.into_iter().collect::<Vec<_>>();
        assert_eq!(target_indices, slice_indices);

        let shape = Shape::new(vec![2, 2, 2]);
        let slice = make_slice(&shape, 2, 1);
        let target_indices = vec![(1, 2), (3, 4), (5, 6), (7, 8)];
        let slice_indices = slice.into_iter().collect::<Vec<_>>();
        assert_eq!(target_indices, slice_indices);
    }

    #[test]
    fn indexing_iterator() {
        let shape = Shape::new(vec![2, 2, 2]);
        let layout = MemoryLayout::RowMajor;
        let tensor_iter: TensorIterator = TensorIterator::new(shape, layout);
        let indices = tensor_iter.into_iter().collect::<Vec<_>>();
        let target_indices = vec![
           vec![1, 0, 0],
           vec![0, 0, 1],
           vec![0, 1, 0],
           vec![0, 1, 1],
           vec![1, 0, 0],
           vec![1, 0, 1],
           vec![1, 1, 0],
           vec![1, 1, 1],
        ]
        .iter()
        .map(|vals| ArrayIndex::new(vals.clone()))
        .collect::<Vec<_>>();
        assert_eq!(indices, target_indices);

        let shape = Shape::new(vec![2, 2, 2]);
        let layout = MemoryLayout::ColumnMajor;
        let tensor_iter: TensorIterator = TensorIterator::new(shape, layout);
        let indices = tensor_iter.into_iter().collect::<Vec<_>>();
        let target_indices = vec![
            vec![0, 0, 0],
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![1, 1, 0],
            vec![0, 0, 1],
            vec![1, 0, 1],
            vec![0, 1, 1],
            vec![1, 1, 1],
        ]
        .iter()
        .map(|vals| ArrayIndex::new(vals.clone()))
        .collect::<Vec<_>>();
        assert_eq!(indices, target_indices);
    }

    #[test]
    fn permuted_tensor_iterator() {
        let shape = Shape::new(vec![2, 2, 2]);
        let permutation = [0, 1, 2];
        let preds = PermutedTensorIterator::new(shape, permutation)
            .into_iter()
            .collect::<Vec<_>>();
        let results = vec![(0, 8)];
        assert_eq!(preds, results);

        let shape = Shape::new(vec![2, 2, 2]);
        let permutation = [1, 0, 2];
        let preds = PermutedTensorIterator::new(shape, permutation)
            .into_iter()
            .collect::<Vec<_>>();
        let results = vec![(0, 2), (4, 6), (2, 4), (6, 8)];
        assert_eq!(preds, results);

        let shape = Shape::new(vec![2, 2, 2]);
        let permutation = [1, 2, 0];
        let preds = PermutedTensorIterator::new(shape, permutation)
            .into_iter()
            .collect::<Vec<_>>();
        let results = vec![
            (0, 1),
            (4, 5),
            (1, 2),
            (5, 6),
            (2, 3),
            (6, 7),
            (3, 4),
            (7, 8),
        ];
        assert_eq!(preds, results);

        let shape = Shape::new(vec![2, 2, 2]);
        let permutation = [2, 1, 0];
        let preds = PermutedTensorIterator::new(shape, permutation)
            .into_iter()
            .collect::<Vec<_>>();
        let results = vec![
            (0, 1),
            (4, 5),
            (2, 3),
            (6, 7),
            (1, 2),
            (5, 6),
            (3, 4),
            (7, 8),
        ];
        assert_eq!(preds, results);
    }
}
