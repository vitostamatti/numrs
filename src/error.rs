use core::fmt;
use std::error::Error;
use std::fmt::Display;

/// ArrayError
#[derive(Debug, Clone)]
pub enum ArrayErrorCode {
    InvalidIndex,
    InvalidAxes,
    InvalidPermutationAxes,
    InvalidPermutationDims,
    InvalidExpandAxes,
    InvalidReshape,
    InvalidPadding,
    InvalidCrop,
    InvalidShape,
}

#[derive(Debug, Clone)]
pub struct ArrayError {
    code: ArrayErrorCode,
}

impl ArrayError {
    pub fn new(code: ArrayErrorCode) -> Self {
        ArrayError { code }
    }
}

impl Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.code)
    }
}

impl Error for ArrayError {}
