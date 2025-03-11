use num_traits::Zero;
use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1e-10; // note: it would be interesting to play around with this.
const COMP_EPSILON: f64 = 1e-10; // note: it would be interesting to play around with this.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderedFloat(f64);

impl OrderedFloat {
    pub fn new(value: f64) -> Self {
        Self(value)
    }
    pub fn numer(&self) -> f64 {
        self.0
    }
    pub fn denom(&self) -> f64 {
        1.0
    }
    pub fn set_zero(&mut self) {
        self.0 = 0.0;
    }

    pub fn recip(&self) -> Self {
        Self::new(1.0 / self.0)
    }
    pub fn new_raw(numer: i32, denom: i32) -> Self {
        Self::new(numer as f64 / denom as f64)
    }

    pub fn is_number(&self) -> bool {
        self.0.is_finite()
    }
    pub fn from_float(value: f64) -> Option<Self> {
        Some(Self::new(value))
    }
    pub fn abs(&self) -> Self {
        Self::new(self.0.abs())
    }
}

// Implement num_traits
impl num_traits::Zero for OrderedFloat {
    fn zero() -> Self {
        Self::new(0.0)
    }
    fn is_zero(&self) -> bool {
        self.0.abs() < EPSILON
    }
}
impl num_traits::One for OrderedFloat {
    fn one() -> Self {
        Self::new(1.0)
    }
    fn is_one(&self) -> bool {
        (self.0 - 1.0).abs() < EPSILON
    }
}
impl num_traits::Signed for OrderedFloat {
    fn is_negative(&self) -> bool {
        !self.is_zero() && self.0.is_sign_negative()
    }
    fn is_positive(&self) -> bool {
        !self.is_zero() && self.0 > 0.0
    }
    fn abs(&self) -> Self {
        Self::new(self.0.abs())
    }
    fn abs_sub(&self, other: &Self) -> Self {
        (self - other).max(OrderedFloat::zero())
    }
    fn signum(&self) -> Self {
        Self::new(self.0.signum())
    }
}
impl num_traits::Num for OrderedFloat {
    type FromStrRadixErr = num_traits::ParseFloatError;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        match f64::from_str_radix(str, radix) {
            Ok(value) => Ok(Self::new(value)),
            Err(err) => Err(err),
        }
    }
}
impl num_traits::FromPrimitive for OrderedFloat {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::new(n as f64))
    }
    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::new(n as f64))
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::new(n))
    }
    fn from_usize(n: usize) -> Option<Self> {
        Some(Self::new(n as f64))
    }
}
impl num_traits::ToPrimitive for OrderedFloat {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 as i64)
    }
    fn to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }
    #[allow(clippy::unnecessary_cast)]
    fn to_f64(&self) -> Option<f64> {
        Some(self.0 as f64)
    }
}

// Implement std ops
impl std::ops::Rem for OrderedFloat {
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        Self::new(self.0 % other.0)
    }
}
impl std::ops::Neg for OrderedFloat {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.0)
    }
}
impl std::ops::Neg for &OrderedFloat {
    type Output = OrderedFloat;
    fn neg(self) -> OrderedFloat {
        OrderedFloat::new(-self.0)
    }
}

// Implement add, sub, mul, div operations, with assign operations, references, by macros
macro_rules! impl_ops {
    ($trait:ident, $method:ident) => {
        impl std::ops::$trait for OrderedFloat {
            type Output = Self;
            fn $method(self, other: Self) -> Self {
                Self::new(self.0.$method(other.0))
            }
        }
        impl std::ops::$trait<&OrderedFloat> for OrderedFloat {
            type Output = Self;
            fn $method(self, other: &Self) -> Self {
                Self::new(self.0.$method(other.0))
            }
        }
        impl std::ops::$trait<OrderedFloat> for &OrderedFloat {
            type Output = OrderedFloat;
            fn $method(self, other: OrderedFloat) -> OrderedFloat {
                OrderedFloat::new(self.0.$method(other.0))
            }
        }
        impl std::ops::$trait<&OrderedFloat> for &OrderedFloat {
            type Output = OrderedFloat;
            fn $method(self, other: &OrderedFloat) -> OrderedFloat {
                OrderedFloat::new(self.0.$method(other.0))
            }
        }
    };
}
impl_ops!(Add, add);
impl_ops!(Sub, sub);
impl_ops!(Mul, mul);
impl_ops!(Div, div);

// Implement assign operations
macro_rules! impl_assign_ops {
        ($trait:ident, $method:ident, $op:tt) => {
            #[allow(clippy::assign_op_pattern)]
            impl std::ops::$trait for OrderedFloat {
                fn $method(&mut self, other: Self) {
                    *self = self.clone() $op other;
                }
            }
            impl std::ops::$trait<&OrderedFloat> for OrderedFloat {
                fn $method(&mut self, other: &Self) {
                    *self = self.clone() $op other;
                }
            }
            // impl std::ops::$trait<&f32> for OrderedFloat {
            //     fn $method(&mut self, other: &f32) {
            //         self.0 = self.0 $op *other as f64;
            //     }
            // }
            // impl std::ops::$trait<&f64> for OrderedFloat {
            //     fn $method(&mut self, other: &f64) {
            //         self.0 = self.0 $op *other as f64;
            //     }
            // }
        };
    }
impl_assign_ops!(AddAssign, add_assign, +);
impl_assign_ops!(SubAssign, sub_assign, -);
impl_assign_ops!(MulAssign, mul_assign, *);
impl_assign_ops!(DivAssign, div_assign, /);

// Implement other std traits
impl std::str::FromStr for OrderedFloat {
    type Err = std::num::ParseFloatError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(f64::from_str(s)?))
    }
}
impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}
impl std::fmt::Display for OrderedFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Implement Eq
impl Eq for OrderedFloat {}

// Implement PartialEq
impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        (self.0 - other.0).abs() < EPSILON
    }
}
impl PartialEq<f64> for OrderedFloat {
    fn eq(&self, other: &f64) -> bool {
        (self.0 - other).abs() < EPSILON
    }
}
impl PartialEq<OrderedFloat> for f64 {
    fn eq(&self, other: &OrderedFloat) -> bool {
        (*self - other.0).abs() < EPSILON
    }
}

// Implement PartialOrd
impl PartialOrd for OrderedFloat {
    #[allow(clippy::non_canonical_partial_ord_impl)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if (self.0 - other.0).abs() < COMP_EPSILON {
            Some(std::cmp::Ordering::Equal)
        } else {
            self.0.partial_cmp(&other.0)
        }
    }
}

// Implement Ord
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if (self.0 - other.0).abs() < COMP_EPSILON {
            std::cmp::Ordering::Equal
        } else {
            self.partial_cmp(other).unwrap()
        }
    }
}

// Implement From<f64> for OrderedFloat
impl From<f64> for OrderedFloat {
    fn from(value: f64) -> Self {
        OrderedFloat::new(value)
    }
}

// Implement Default
impl Default for OrderedFloat {
    fn default() -> Self {
        Self::new(0.0)
    }
}

// Implement Sum for OrderedFloat
impl std::iter::Sum for OrderedFloat {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), std::ops::Add::add)
    }
}

// Implement Sum for references to OrderedFloat
impl<'a> std::iter::Sum<&'a OrderedFloat> for OrderedFloat {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, item| acc + item)
    }
}

// comparisons using references
impl PartialEq<&OrderedFloat> for OrderedFloat {
    fn eq(&self, other: &&Self) -> bool {
        (self.0 - other.0).abs() < EPSILON
    }
}

impl PartialEq<OrderedFloat> for &OrderedFloat {
    fn eq(&self, other: &OrderedFloat) -> bool {
        (self.0 - other.0).abs() < EPSILON
    }
}

// impl PartialEq<&OrderedFloat> for &OrderedFloat {
//     fn eq(&self, other: &&OrderedFloat) -> bool {
//         (self.0 - other.0).abs() < EPSILON
//     }
// }

impl PartialOrd<&OrderedFloat> for OrderedFloat {
    fn partial_cmp(&self, other: &&Self) -> Option<std::cmp::Ordering> {
        if (self.0 - other.0).abs() < COMP_EPSILON {
            Some(std::cmp::Ordering::Equal)
        } else {
            self.0.partial_cmp(&other.0)
        }
    }
}

impl PartialOrd<OrderedFloat> for &OrderedFloat {
    fn partial_cmp(&self, other: &OrderedFloat) -> Option<std::cmp::Ordering> {
        if (self.0 - other.0).abs() < COMP_EPSILON {
            Some(std::cmp::Ordering::Equal)
        } else {
            self.0.partial_cmp(&other.0)
        }
    }
}

// impl PartialOrd<&OrderedFloat> for &OrderedFloat {
//     fn partial_cmp(&self, other: &&OrderedFloat) -> Option<std::cmp::Ordering> {
//         if (self.0 - other.0).abs() < EPSILON {
//             Some(std::cmp::Ordering::Equal)
//         } else {
//             self.0.partial_cmp(&other.0)
//         }
//     }
// }

// impl Ord for &OrderedFloat {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.partial_cmp(other).unwrap()
//     }
// }
