use crate::neurlang::{ArrayIndex, MemoryLayout, Shape};
use crate::utils::count_elements;

// //////////////////
// Slicing
// //////////////////
pub fn make_slice(shape: &Shape, axis: usize, index: usize) -> Slice {
    Slice { shape, axis, index }
}

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
        let n_prefix = count_elements(&self.shape.dimensions[0..axis]);
        let n_axis_suffix = count_elements(&self.shape.dimensions[axis..]);
        let n_suffix = count_elements(&self.shape.dimensions[(axis + 1)..]);
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
pub struct TensorIterator<const N: usize> {
    shape: Shape,
    memory_layout: MemoryLayout,
    return_index: [usize; N],
    count: usize,
    max_count: usize,
}

impl<const N: usize> TensorIterator<N> {
    pub fn new(shape: Shape, memory_layout: MemoryLayout) -> TensorIterator<N> {
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
    type Item = ArrayIndex<N>;

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
                    let mut idx = N - 1;
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
        let tensor_iter: TensorIterator<3> = TensorIterator::new(shape, layout);
        let indices = tensor_iter.into_iter().collect::<Vec<_>>();
        let target_indices = vec![
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
        .iter()
        .map(|&vals| ArrayIndex::new(vals))
        .collect::<Vec<_>>();
        assert_eq!(indices, target_indices);

        let shape = Shape::new(vec![2, 2, 2]);
        let layout = MemoryLayout::ColumnMajor;
        let tensor_iter: TensorIterator<3> = TensorIterator::new(shape, layout);
        let indices = tensor_iter.into_iter().collect::<Vec<_>>();
        let target_indices = vec![
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        .iter()
        .map(|&vals| ArrayIndex::new(vals))
        .collect::<Vec<_>>();
        assert_eq!(indices, target_indices);
    }
}
