//! Pointer Types
//!

use crate::parking_lot::lock_api::{RwLockReadGuard, RwLockWriteGuard};
use crate::parking_lot::{RawRwLock, RwLock};
use std::sync::{Arc, Weak};

pub trait RwLockPtr<ObjType> {
    fn new_ptr(ptr: Arc<RwLock<ObjType>>) -> Self;

    fn new_value(obj: ObjType) -> Self;

    fn ptr(&self) -> &Arc<RwLock<ObjType>>;

    fn ptr_mut(&mut self) -> &mut Arc<RwLock<ObjType>>;

    #[inline(always)]
    fn read_recursive(&self) -> RwLockReadGuard<RawRwLock, ObjType> {
        let ret = self.ptr().read_recursive();
        ret
    }

    #[inline(always)]
    fn write(&self) -> RwLockWriteGuard<RawRwLock, ObjType> {
        let ret = self.ptr().write();
        ret
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.ptr(), other.ptr())
    }
}

pub struct ArcRwLock<T> {
    ptr: Arc<RwLock<T>>,
}

pub struct WeakRwLock<T> {
    ptr: Weak<RwLock<T>>,
}

impl<T> ArcRwLock<T> {
    pub fn downgrade(&self) -> WeakRwLock<T> {
        WeakRwLock::<T> {
            ptr: Arc::downgrade(&self.ptr),
        }
    }
}

impl<T> WeakRwLock<T> {
    pub fn upgrade_force(&self) -> ArcRwLock<T> {
        ArcRwLock::<T> {
            ptr: self.ptr.upgrade().unwrap(),
        }
    }
    pub fn upgrade(&self) -> Option<ArcRwLock<T>> {
        self.ptr.upgrade().map(|x| ArcRwLock::<T> { ptr: x })
    }
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.ptr, &other.ptr)
    }
}

impl<T: Send + Sync> Clone for ArcRwLock<T> {
    fn clone(&self) -> Self {
        Self::new_ptr(Arc::clone(self.ptr()))
    }
}

impl<T: Send + Sync> RwLockPtr<T> for ArcRwLock<T> {
    fn new_ptr(ptr: Arc<RwLock<T>>) -> Self {
        Self { ptr }
    }
    fn new_value(obj: T) -> Self {
        Self::new_ptr(Arc::new(RwLock::new(obj)))
    }
    #[inline(always)]
    fn ptr(&self) -> &Arc<RwLock<T>> {
        &self.ptr
    }
    #[inline(always)]
    fn ptr_mut(&mut self) -> &mut Arc<RwLock<T>> {
        &mut self.ptr
    }
}

impl<T: Send + Sync> PartialEq for ArcRwLock<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
    }
}

impl<T: Send + Sync> Eq for ArcRwLock<T> {}

impl<T> Clone for WeakRwLock<T> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr.clone() }
    }
}

impl<T> PartialEq for WeakRwLock<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr.ptr_eq(&other.ptr)
    }
}

impl<T> Eq for WeakRwLock<T> {}

impl<T> std::ops::Deref for ArcRwLock<T> {
    type Target = RwLock<T>;
    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Tester {
        idx: usize,
    }

    type TesterPtr = ArcRwLock<Tester>;
    type TesterWeak = WeakRwLock<Tester>;

    impl std::fmt::Debug for TesterPtr {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            let value = self.read_recursive();
            write!(f, "{:?}", value)
        }
    }

    impl std::fmt::Debug for TesterWeak {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            self.upgrade_force().fmt(f)
        }
    }

    #[test]
    fn pointers_test_1() {
        // cargo test pointers_test_1 -- --nocapture
        let ptr = TesterPtr::new_value(Tester { idx: 0 });
        let weak = ptr.downgrade();
        ptr.write().idx = 1;
        assert_eq!(weak.upgrade_force().read_recursive().idx, 1);
        weak.upgrade_force().write().idx = 2;
        assert_eq!(ptr.read_recursive().idx, 2);
    }
}
