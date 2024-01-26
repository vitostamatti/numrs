use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use crate::Num;

#[derive(Debug, Clone)]
pub struct Array<T> {
    // continuous buffer of data
    data: Rc<Vec<T>>,
    /// stores the length of each dimension
    shape: Vec<usize>,
    /// stores the offset in the buffer between successive elements in that dimension
    strides: Vec<usize>,
    /// strides offset
    offset: usize,
}

// Properties
impl<T: Copy + Num> Array<T> {
    /// Creates a new array with the given data and shape
    pub fn new(data: &[T], shape: &[usize]) -> Self {
        Self {
            data: Rc::new(data.into()),
            shape: shape.into(),
            strides: Self::contiguous_strides(shape),
            offset: 0,
        }
    }

    /// Creates a new 1d array with the given value
    pub fn scalar(value: T) -> Self {
        Self::new(&[value], &[1])
    }

    /// Creates a new array with the given shape, and fill it with the given value.
    pub fn constant(value: T, shape: &[usize]) -> Self {
        Self::new(&[value], &vec![1; shape.len()]).expand(shape)
    }

    pub fn eye(dim: usize) -> Self {
        Self::scalar(T::ONE)
            .pad(&[(0, dim)])
            .reshape(&[1, dim + 1])
            .expand(&[dim, dim + 1])
            .reshape(&[dim * (dim + 1)])
            .crop(&[(0, dim * dim)])
            .reshape(&[dim, dim])
    }

    pub fn ndims(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        if self.ndims() == 0 {
            0
        } else {
            self.shape.iter().product()
        }
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// This function returns the corresponding element index
    /// of the ArrayBuffer given the provided index  
    pub fn index(&self, at: &[usize]) -> usize {
        self.offset
            + at.iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>()
    }

    fn is_valid_index(&self, index: &[usize]) -> bool {
        let is_empty = self.shape.is_empty();
        let eq_dims = self.ndims() == index.len();
        let idx_lower_than_shape = index.iter().zip(self.shape.iter()).all(|(i, s)| i < s);
        !is_empty && eq_dims && idx_lower_than_shape
    }
    fn is_valid_reduce(&self, axes: &[usize]) -> bool {
        axes.iter().all(|a| *a < self.shape.len())
    }
    fn is_valid_permute_axes(&self, axes: &[usize]) -> bool {
        axes.len() * (axes.len() - 1) / 2 == axes.iter().sum()
    }
    fn is_valid_permute_dims(&self, axes: &[usize]) -> bool {
        axes.iter().all(|x| *x < self.ndims())
    }
    fn is_valid_expand(&self, shape: &[usize]) -> bool {
        self.ndims() <= shape.len()
    }
    fn is_valid_reshape(&self, shape: &[usize]) -> bool {
        self.size() == shape.iter().product()
    }
    fn is_valid_pad(&self, padding: &[(usize, usize)]) -> bool {
        padding.len() == self.ndims()
    }
    fn is_valid_crop(&self, limits: &[(usize, usize)]) -> bool {
        if limits.len() != self.ndims() {
            return false;
        }

        for (i, &(start, end)) in limits.iter().enumerate() {
            if start >= end {
                return false;
            }
            if end > self.shape[i] {
                return false;
            }
        }
        return true;
    }
    /// Returns the value at the given index. There must be as many indices as there are dimensions.
    pub fn at(&self, index: &[usize]) -> T {
        let limits = index.iter().map(|&i| (i, i + 1)).collect::<Vec<_>>();
        self.crop(&limits).ravel()[0]
    }

    /// Creates a new ArrayShaper striding to
    /// a contiguous buffer in row-first order.
    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        strides
    }

    fn with_shape_and_strides(&self, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self {
            data: Rc::clone(&self.data),
            shape,
            strides,
            offset: self.offset,
        }
    }

    fn with_shape_and_offset(&self, shape: Vec<usize>, offset: usize) -> Self {
        Self {
            data: Rc::clone(&self.data),
            shape: shape,
            strides: self.strides.clone(),
            offset: offset,
        }
    }

    fn with_data(&self, data: &[T]) -> Self {
        Self::new(data, self.shape())
    }

    /// Apply the given shape permutation.
    pub fn permute(&self, axes: &[usize]) -> Self {
        // if !self.is_valid_permute_axes(axes) {
        //     panic!("Cannot permute on axes");
        // }
        // if !self.is_valid_permute_dims(axes) {
        //     panic!("Cannot permute on axes because of dims");
        // }
        let mut shape = Vec::with_capacity(self.ndims());
        let mut strides = Vec::with_capacity(self.ndims());
        for &axis in axes {
            shape.push(self.shape[axis]);
            strides.push(self.strides[axis]);
        }
        self.with_shape_and_strides(shape, strides)
    }

    /// Expand dimensions
    pub fn expand(&self, shape: &[usize]) -> Self {
        // if !self.is_valid_expand(shape) {
        //     panic!("Expand operations is not possible");
        // }
        let ndims = shape.len();
        let mut new_shape = Vec::with_capacity(ndims);
        let mut new_strides = Vec::with_capacity(ndims);
        for (from_dim, to_dim) in (0..self.ndims()).rev().zip((0..ndims).rev()) {
            if self.shape[from_dim] == shape[to_dim] {
                new_shape.push(self.shape[from_dim]);
                new_strides.push(self.strides[from_dim]);
            } else if self.shape[from_dim] == 1 {
                new_shape.push(shape[to_dim]);
                new_strides.push(0);
            }
        }
        new_shape.reverse();
        new_strides.reverse();

        self.with_shape_and_strides(new_shape, new_strides)
    }

    /// Remove all dimensions of length 1.
    pub fn squeeze(&self) -> Self {
        let mut shape = Vec::with_capacity(self.ndims());
        let mut strides = Vec::with_capacity(self.ndims());
        for (&dim, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if dim != 1 {
                shape.push(dim);
                strides.push(stride);
            }
        }
        self.with_shape_and_strides(shape, strides)
    }

    /// Adds padding after and before the given axis.
    pub fn pad(&self, padding: &[(usize, usize)]) -> Self {
        // if !self.is_valid_pad(padding) {
        //     panic!("Padding is not possible");
        // }
        let mut shape = self.shape.clone();
        for (i, &(before, after)) in padding.iter().enumerate() {
            shape[i] = self.shape[i] + before + after;
        }
        let strides = Self::contiguous_strides(&shape);
        self.with_shape_and_strides(shape, strides)
    }

    /// Crops before and after for the given axis;
    pub fn crop(&self, limits: &[(usize, usize)]) -> Self {
        // if !self.is_valid_crop(limits) {
        //     panic!("Crop operation is not possible");
        // }
        let index = limits.iter().map(|&(start, _)| start).collect::<Vec<_>>();
        let offset = self.index(&index);
        let shape = limits.iter().map(|&(start, end)| end - start).collect();
        self.with_shape_and_offset(shape, offset)
    }

    pub fn reshape(&self, shape: &[usize]) -> Self {
        if self.size() != shape.iter().product() {
            panic!("Cannot reshape");
        }
        let new_ndims = shape.len();
        let mut new_strides = vec![0; new_ndims];

        let squeezed = self.squeeze();

        let old_shape = &squeezed.shape;
        let old_ndims = squeezed.ndims();
        let old_strides = &squeezed.strides;

        let (mut oi, mut oj) = (0, 1);
        let (mut ni, mut nj) = (0, 1);

        while (ni < new_ndims) && (oi < old_ndims) {
            // First find the dimensions in both old and new we can combine -
            // by checking that the number of elements in old and new is the same.
            let mut np = shape[ni];
            let mut op = old_shape[oi];

            while np != op {
                if np < op {
                    np *= shape[nj];
                    nj += 1;
                } else {
                    op *= old_shape[oj];
                    oj += 1;
                }
            }

            for ok in oi..oj - 1 {
                if old_strides[ok] != old_strides[ok + 1] * old_shape[ok + 1] {
                    panic!("Could not reshape array");
                }
            }

            new_strides[nj - 1] = old_strides[oj - 1];
            for nk in (ni + 1..nj).rev() {
                new_strides[nk - 1] = new_strides[nk] * shape[nk];
            }

            ni = nj;
            nj += 1;
            oi = oj;
            oj += 1;
        }

        let last_stride = if ni >= 1 { new_strides[ni - 1] } else { 1 };
        for stride in new_strides.iter_mut().take(new_ndims).skip(ni) {
            *stride = last_stride;
        }
        self.with_shape_and_strides(shape.to_vec(), new_strides)
    }

    pub fn broadcast(&self, shape: &[usize]) -> Self {
        if shape.len() < self.ndims() {
            panic!("Array broadcast not possible");
        }
        let added_dims = shape.len() - self.ndims();
        let mut strides = vec![0; added_dims];
        for (&dst_dim, (&src_dim, &src_stride)) in shape[added_dims..]
            .iter()
            .zip(self.shape.iter().zip(&self.strides))
        {
            let s = if dst_dim == src_dim {
                src_stride
            } else if src_dim != 1 {
                panic!("Incompatible broadcast");
            } else {
                0
            };
            strides.push(s)
        }
        self.with_shape_and_strides(shape.to_vec(), strides)
    }

    fn broadcast_binary_op(&self, other: &Self, f: impl Fn(&Self, &Self) -> Self) -> Self {
        if self.ndims() > other.ndims() {
            // Rust tidbit: I originally did not have a reverse parameter,
            // but just called |a,b| f(b,a) in the recursive call. This doesn't work,
            // because it hits the recursion limit: https://stackoverflow.com/questions/54613966/error-reached-the-recursion-limit-while-instantiating-funcclosure
            panic!("Cannot broadcast and apply operation");
        }

        if self.ndims() == other.ndims() {
            let res_shape = self
                .shape()
                .iter()
                .zip(other.shape().iter())
                .map(|(a, b)| *a.max(b))
                .collect::<Vec<_>>();
            let s_expanded = self.expand(&res_shape);
            let o_expanded = other.expand(&res_shape);
            return f(&s_expanded, &o_expanded);
        }

        let num_ones_to_add = other.shape().len().saturating_sub(self.shape().len());
        let mut new_shape = vec![1; num_ones_to_add];
        new_shape.extend(self.shape());

        self.reshape(&new_shape).broadcast_binary_op(other, f)
    }

    fn binary_op(&self, other: &Self, f: impl Fn(T, T) -> T) -> Self {
        // if !self.shaper.is_valid_zip(&other.shaper) {
        //     panic!("Zip operation is not valid");
        // }
        let mut result: Vec<_> = self.into_iter().collect();
        for (x, y) in result.iter_mut().zip(other.into_iter()) {
            *x = f(*x, y);
        }
        self.with_data(&result)
    }

    // -------------- Binary Ops --------------
    pub fn add(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| a.binary_op(b, T::add))
    }

    pub fn mul(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| a.binary_op(b, T::mul))
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| a.binary_op(b, T::sub))
    }

    pub fn div(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| a.binary_op(b, T::div))
    }
    /// Raise self to the power of other, element-wise.
    pub fn pow(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| a.binary_op(b, T::powf))
    }

    // ------------- Ord and Eq Ops ---------------
    pub fn eq(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| {
            a.binary_op(b, |x, y| if x == y { T::ONE } else { T::ZERO })
        })
    }
    pub fn lt(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| {
            a.binary_op(b, |x, y| if x < y { T::ONE } else { T::ZERO })
        })
    }
    pub fn gt(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| {
            a.binary_op(b, |x, y| if x > y { T::ONE } else { T::ZERO })
        })
    }
    pub fn lte(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| {
            a.binary_op(b, |x, y| if x <= y { T::ONE } else { T::ZERO })
        })
    }
    pub fn gte(&self, other: &Self) -> Self {
        self.broadcast_binary_op(other, |a, b| {
            a.binary_op(b, |x, y| if x >= y { T::ONE } else { T::ZERO })
        })
    }
    // ---------------------------------------

    /// returns the contiguous data inside as a vector
    pub fn ravel(&self) -> Vec<T> {
        self.into_iter().collect()
    }
    fn unary_op(&self, f: impl Fn(T) -> T) -> Self {
        let mut result = self.ravel();
        for elem in &mut result {
            *elem = f(*elem);
        }
        self.with_data(&result)
    }
    // ------------- Unary Ops ---------------
    /// Apply the natural logarithm to each element.
    pub fn log(&self) -> Self {
        self.unary_op(T::log)
    }
    /// Apply exp to each element.
    pub fn exp(&self) -> Self {
        self.unary_op(T::exp)
    }
    // ---------------------------------------

    fn reduce_shape(&self, axes: &[usize]) -> Vec<usize> {
        let mut shape = self.shape.clone();
        for &axis in axes {
            shape[axis] = 1;
        }
        shape
    }

    fn reduce_strides(&self, axes: &[usize], shape: &[usize]) -> Vec<usize> {
        let mut strides = Self::contiguous_strides(shape);
        for &axis in axes {
            strides[axis] = 0;
        }
        strides
    }

    fn reduce_on_axes(&self, axes: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let shape = self.reduce_shape(axes);
        let strides = self.reduce_strides(axes, &shape);
        (shape, strides)
    }

    fn iter_index(&self) -> IndexIterator<T> {
        IndexIterator::new(self)
    }

    pub fn reduce(&self, initial: T, f: impl Fn(T, T) -> T, axes: &[usize]) -> Self {
        let (shape, strides) = self.reduce_on_axes(axes);

        let reducer = self.with_shape_and_strides(shape.clone(), strides);

        let mut data = vec![initial; self.size()];

        for idx in self.iter_index() {
            let index = self.index(&idx);
            let result_index = reducer.index(&idx);
            data[result_index] = f(data[result_index], self.data[index]);
        }
        Self::new(&data, &shape)
    }

    // ------------- Reduce Ops ---------------
    pub fn sum(&self, axes: &[usize]) -> Self {
        self.reduce(T::ZERO, T::add, axes)
    }

    pub fn max(&self, axes: &[usize]) -> Self {
        self.reduce(T::MIN, |x, y| if x > y { x } else { y }, axes)
    }

    pub fn min(&self, axes: &[usize]) -> Self {
        self.reduce(T::MAX, |x, y| if x < y { x } else { y }, axes)
    }
    // ----------------------------------------

    // -------------- Linalg Ops --------------

    /// Switch the two axes around.
    pub fn transpose(&self, axis0: usize, axis1: usize) -> Self {
        let mut axes = (0..self.ndims()).collect::<Vec<_>>();
        axes.swap(axis0, axis1);
        self.permute(&axes)
    }

    /// Matrix multiplication, generalized to tensors.
    /// i.e. multiply [..., m, n] with [..., n, o] to [..., m, o]
    pub fn matmul(&self, other: &Self) -> Self {
        // self's shape from [..., m, n] to [..., m, 1, n]
        // using just reshape.
        let s = self.shape();
        let self_shape = [&s[..s.len() - 1], &[1, s[s.len() - 1]]].concat();
        let l = self.reshape(&self_shape);

        // other's shape from [..., n, o] to [..., 1, o, n]
        // using reshape + transpose.
        let s = other.shape();
        let other_shape = [&s[..s.len() - 2], &[1], &s[s.len() - 2..]].concat();
        let r = other
            .reshape(&other_shape)
            .transpose(other_shape.len() - 1, other_shape.len() - 2);

        // after multiply: [..., m, o, n]
        let prod = &l * &r;
        // after sum:      [..., m, o, 1]
        let sum = prod.sum(&[prod.shape().len() - 1]);
        // after reshape:  [..., m, o]
        let s = sum.shape();
        sum.reshape(&s[..s.len() - 1])
    }
    // ----------------------------------------
}

pub struct IndexIterator<'a, T> {
    array: &'a Array<T>,
    index: Vec<usize>,
    valid: bool,
}

impl<'a, T: Copy + Num> IndexIterator<'a, T> {
    pub fn new(array: &'a Array<T>) -> Self {
        let index = vec![0; array.ndims()];
        // let valid = array.is_valid_index(&index);
        let valid = true;
        Self {
            array,
            index,
            valid,
        }
    }
}

impl<'a, T: Copy + Num> Iterator for IndexIterator<'a, T> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.valid {
            return None;
        }

        let result = self.index.clone();

        for i in (0..self.array.ndims()).rev() {
            self.index[i] += 1;
            if self.index[i] < self.array.shape[i] {
                break;
            }
            self.index[i] = 0;
        }

        self.valid = !self.index.iter().all(|e| *e == 0);

        Some(result)
    }
}

pub struct ArrayIterator<'a, T> {
    index_iterator: IndexIterator<'a, T>,
}

impl<'a, T: Copy + Num> Iterator for ArrayIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iterator
            .next()
            .map(|index| self.index_iterator.array.data[self.index_iterator.array.index(&index)])
    }
}

impl<'a, T: Copy + Num> IntoIterator for &'a Array<T> {
    type Item = T;
    type IntoIter = ArrayIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        ArrayIterator {
            index_iterator: IndexIterator::new(self),
        }
    }
}

macro_rules! impl_ops_array {
    ($op_trait:ident, $op_fn:ident) => {
        impl<T: Num> $op_trait<&Array<T>> for &Array<T> {
            type Output = Array<T>;

            fn $op_fn(self, rhs: &Array<T>) -> Array<T> {
                Array::<_>::$op_fn(self, rhs)
            }
        }

        impl<T: Num> $op_trait<&Array<T>> for Array<T> {
            type Output = Array<T>;

            fn $op_fn(self, rhs: &Array<T>) -> Array<T> {
                Self::$op_fn(&self, rhs)
            }
        }

        impl<T: Num> $op_trait<Array<T>> for Array<T> {
            type Output = Array<T>;

            fn $op_fn(self, rhs: Array<T>) -> Array<T> {
                Self::$op_fn(&self, &rhs)
            }
        }

        impl<T: Num> $op_trait<Array<T>> for &Array<T> {
            type Output = Array<T>;

            fn $op_fn(self, rhs: Array<T>) -> Array<T> {
                Self::$op_fn(self, &rhs)
            }
        }
    };
}
impl_ops_array!(Add, add);
impl_ops_array!(Sub, sub);
impl_ops_array!(Mul, mul);
impl_ops_array!(Div, div);

impl<T: Num> Neg for &Array<T> {
    type Output = Array<T>;

    fn neg(self) -> Array<T> {
        Array::constant(T::ZERO, &self.shape).sub(self)
    }
}

impl<T: Num> Neg for Array<T> {
    type Output = Array<T>;

    fn neg(self) -> Array<T> {
        Array::constant(T::ZERO, &self.shape).sub(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;

    #[test]
    fn test_new_array() {
        let shape = &[2, 2];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);
        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array.data(), &[2.0, 1.0, 4.0, 2.0]);
    }

    #[test]
    fn test_contiguous() {
        let shape = &[2, 2];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);
        assert_eq!(array.strides, [2, 1].to_vec());
        assert_eq!(array.offset, 0);
    }

    #[test]
    fn test_index() {
        let shape = &[2, 2];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);

        let index0 = array.index(&[0, 0]);
        assert_eq!(index0, 0);

        let index1 = array.index(&[1, 1]);
        assert_eq!(index1, 3);

        let index2 = array.index(&[0, 1]);
        assert_eq!(index2, 1);
    }

    #[test]
    fn test_at() {
        let shape = &[2, 2];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);

        let value = array.at(&[0, 0]);
        assert_eq!(value, 2.0);
        let value = array.at(&[1, 1]);
        assert_eq!(value, 2.0);
        let value = array.at(&[1, 0]);
        assert_eq!(value, 4.0);
    }

    #[test]
    fn test_squeeze() {
        let shape = &[1, 1, 2, 2];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);
        let arrra1 = array.squeeze();
        assert_eq!(arrra1.shape, &[2, 2]);
    }

    #[test]
    fn test_reshape() {
        let shape = &[2, 2];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);

        let shape1 = &[1, 4];
        let array1 = array.reshape(shape1);

        assert_eq!(array1.shape, &[1, 4]);
        assert_eq!(array1.strides, &[4, 1]);

        let shape2 = &[4, 1];
        let array2 = array.reshape(shape2);

        assert_eq!(array2.shape, &[4, 1]);
        assert_eq!(array2.strides, &[1, 1]);

        let shape3 = &[1, 4, 1, 1];
        let array3 = array.reshape(shape3);

        assert_eq!(array3.shape, &[1, 4, 1, 1]);
        assert_eq!(array3.strides, &[4, 1, 1, 1]);
    }

    #[test]
    fn test_permute() {
        let shape = &[1, 4];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);
        let array1 = array.permute(&[1, 0]);
        assert_eq!(array1.shape, &[4, 1]);
    }

    #[test]
    fn test_transpose() {
        let shape = &[2, 2];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);
        let array1 = array.transpose(0, 1);
        assert_eq!(array1.strides, &[1, 2]);
    }

    #[test]
    fn test_pad() {
        let shape = &[2, 2];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);
        let array1 = array.pad(&[(1, 1), (1, 1)]);
        assert_eq!(array1.shape, &[4, 4]);
        assert_eq!(array1.strides, &[4, 1]);
    }

    // crop
    #[test]
    fn test_crop() {
        let shape = &[3, 3];
        let data = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let array = Array::new(data, shape);
        let array1 = array.crop(&[(1, 3), (0, 2)]);
        assert_eq!(array1.shape, &[2, 2]);
        assert_eq!(array1.offset, 3);
        assert_eq!(array1.strides, &[3, 1]);
    }

    #[test]
    fn test_broadcast() {
        let shape = &[1, 4];
        let data = &[2.0, 1.0, 4.0, 2.0];
        let array = Array::new(data, shape);

        let array1 = array.broadcast(&[4, 4]);
        assert_eq!(array1.shape(), &[4, 4])
    }

    #[test]
    fn test_unary_ops() {
        let shape = &[2, 2];
        let data = &[2.0, 2.0, 2.0, 2.0];
        let array = &Array::new(data, shape);

        let array1 = array.log();
        for d in array1.data().iter() {
            assert!(d - 0.6931 < 0.0001)
        }

        let array2 = array.exp();
        for d in array2.data().iter() {
            assert!(d - 7.3890 < 0.0001)
        }
    }

    #[test]
    fn test_binary_ops() {
        let shape = &[2, 2];
        let data = &[1.0, 1.0, 1.0, 1.0];
        let array1 = &Array::new(data, shape);

        let shape = &[2, 2];
        let data = &[2.0, 2.0, 2.0, 2.0];
        let array2 = &Array::new(data, shape);

        // Add
        let array3 = array1.add(array2);
        assert_eq!(array3.data(), [3.0, 3.0, 3.0, 3.0]);

        // Sub
        let array3 = array1.sub(array2);
        assert_eq!(array3.data(), &[-1.0, -1.0, -1.0, -1.0]);

        // Mul
        let array3 = array1.mul(array2);
        assert_eq!(array3.data(), &[2.0, 2.0, 2.0, 2.0]);

        // Div
        let array3 = array1.div(array2);
        assert_eq!(array3.data(), &[0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_binary_ops_traits() {
        let shape = &[2, 2];
        let data = &[1.0, 1.0, 1.0, 1.0];
        let array1 = &Array::new(data, shape);

        let shape = &[2, 2];
        let data = &[2.0, 2.0, 2.0, 2.0];
        let array2 = &Array::new(data, shape);

        // Add
        let array3 = array1 + array2;
        assert_eq!(array3.data(), [3.0, 3.0, 3.0, 3.0]);

        // Sub
        let array3 = array1 - array2;
        assert_eq!(array3.data(), &[-1.0, -1.0, -1.0, -1.0]);

        // Mul
        let array3 = array1 * array2;
        assert_eq!(array3.data(), &[2.0, 2.0, 2.0, 2.0]);

        // Div
        let array3 = array1 / array2;
        assert_eq!(array3.data(), &[0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_reduce_ops() {
        let shape = &[2, 2];
        let data = &[1.0, 2.0, 3.0, 4.0];
        let array = &Array::new(data, shape);
        // Max
        let array1 = array.max(&[0, 1]);
        assert_eq!(array1.shape(), &[1, 1]);
        assert_eq!(array1.data[0], 4.0);
        // Min
        let array1 = array.min(&[0, 1]);
        assert_eq!(array1.shape(), &[1, 1]);
        assert_eq!(array1.data[0], 1.0);
        // Sum
        let array1 = array.sum(&[0, 1]);
        assert_eq!(array1.shape(), &[1, 1]);
        assert_eq!(array1.data[0], 10.0);
    }

    #[test]
    fn test_matmul_square() {
        let shape = &[2, 2];
        let data = &[2.0, 2.0, 2.0, 2.0];
        let array1 = &Array::new(data, shape);

        let shape = &[2, 2];
        let data = &[2.0, 2.0, 2.0, 2.0];
        let array2 = &Array::new(data, shape);

        let array3 = array1.matmul(array2);

        assert_eq!(array1.shape[1], array3.shape[0]);
        assert_eq!(array2.shape[0], array3.shape[1]);
        assert_eq!(array3.shape, &[2, 2]);
    }

    #[test]
    fn test_matmul_non_square() {
        let shape = &[2, 3];
        let data = &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let array1 = &Array::new(data, shape);

        let shape = &[3, 2];
        let data = &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let array2 = &Array::new(data, shape);

        let array3 = array1.matmul(array2);
        assert_eq!(array3.shape, &[2, 2]);
        assert_eq!(array3.ravel(), &[12.0, 12.0, 12.0, 12.0]);
    }

    #[test]
    fn test_matmul_broadcast() {
        let shape = &[1];
        let data = &[2.0];
        let array1 = &Array::new(data, shape);

        let shape = &[2, 3];
        let data = &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let array2 = &Array::new(data, shape);

        let array3 = array1.matmul(array2);
        assert_eq!(array3.shape, &[1, 3]);
        assert_eq!(array3.ravel(), &[8.0, 8.0, 8.0]);
    }

    #[test]
    fn test_ord() {
        let shape = &[2, 2];
        let data = &[2.1, 1.0, 1.0, 1.0];
        let array1 = &Array::new(data, shape);

        let shape = &[2, 2];
        let data = &[2.0, 1.0, 1.0, 0.9];
        let array2 = &Array::new(data, shape);

        let array3 = array1.eq(array2);
        assert_eq!(array3.ravel(), &[0.0, 1.0, 1.0, 0.0]);

        let array3 = array1.lt(array2);
        assert_eq!(array3.ravel(), &[0.0, 0.0, 0.0, 0.0]);

        let array3 = array1.gt(array2);
        assert_eq!(array3.ravel(), &[1.0, 0.0, 0.0, 1.0]);

        let array3 = array1.lte(array2);
        assert_eq!(array3.ravel(), &[0.0, 1.0, 1.0, 0.0]);

        let array3 = array1.gte(array2);
        assert_eq!(array3.ravel(), &[1.0, 1.0, 1.0, 1.0]);
    }
}

// macro_rules! assert_delta {
//     ($x:expr, $y:expr, $d:expr) => {
//         if !($x - $y < $d || $y - $x < $d) {
//             panic!();
//         }
//     };
// }
