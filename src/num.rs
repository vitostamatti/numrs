use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

// Number
pub trait Num:
    Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + PartialEq
    + PartialOrd
    + Copy
    + Debug
{
    const MIN: Self;
    const MAX: Self;
    const ZERO: Self;
    const ONE: Self;

    /// Apply exponential function.
    fn exp(self) -> Self;
    /// Apply the natural logarithm.
    fn log(self) -> Self;
    /// Raise self to the power of given exponent.
    fn powf(self, exponent: Self) -> Self;
}
impl Num for f64 {
    const MIN: Self = f64::MIN;
    const MAX: Self = f64::MAX;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn exp(self) -> Self {
        self.exp()
    }

    fn log(self) -> Self {
        self.log(std::f64::consts::E)
    }

    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
}
impl Num for f32 {
    const MIN: Self = f32::MIN;
    const MAX: Self = f32::MAX;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn exp(self) -> Self {
        self.exp()
    }

    fn log(self) -> Self {
        self.log(std::f32::consts::E)
    }

    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
}
