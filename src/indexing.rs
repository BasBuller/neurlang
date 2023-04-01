use crate::neurlang::MemoryLayout;
use crate::array::{ArrayIndex, Shape};

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
}
