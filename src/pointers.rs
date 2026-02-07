//! Pointer Types
//!
//! This module provides a unified interface for reference-counted objects with interior mutability.
//! 
//! Feature Flags:
//! - Default: Uses `Arc<RwLock<T>>` (Safe, Thread-safe).
//! - `unsafe_pointer`: Uses `Arc<UnsafeCell<T>>` (Unsafe, High-performance, Identity-based).

use std::sync::{Arc, Weak};

// =========================================================================================
//  UNSAFE IMPLEMENTATION (feature = "unsafe_pointer")
//  Uses UnsafeCell for maximum performance, bypassing locks.
// =========================================================================================
cfg_if::cfg_if! {
    if #[cfg(feature="unsafe_pointer")] {
        use std::cell::UnsafeCell;

        /// Trait for unsafe pointer operations.
        /// WARNING: The user must ensure no data races occur.
        pub trait UnsafePtr<ObjType> {
            fn new_ptr(ptr: Arc<UnsafeCell<ObjType>>) -> Self;
            fn new_value(obj: ObjType) -> Self;
            
            fn ptr(&self) -> &Arc<UnsafeCell<ObjType>>;
            fn ptr_mut(&mut self) -> &mut Arc<UnsafeCell<ObjType>>;

            #[inline(always)]
            fn read_recursive(&self) -> &ObjType {
                // SAFETY: User promises no concurrent mutable access.
                unsafe { &*self.ptr().get() }
            }

            #[inline(always)]
            fn write(&self) -> &mut ObjType {
                // SAFETY: User promises uniqueness.
                unsafe { &mut *self.ptr().get() }
            }
            
            #[inline(always)]
            fn try_write(&self) -> Option<&mut ObjType> {
                Some(self.write())
            }

            fn ptr_eq(&self, other: &Self) -> bool {
                Arc::ptr_eq(self.ptr(), other.ptr())
            }
        }

        pub struct ArcUnsafe<T> {
            ptr: Arc<UnsafeCell<T>>,
        }

        pub struct WeakUnsafe<T> {
            ptr: Weak<UnsafeCell<T>>,
        }

        // SAFETY: We manually implement Send/Sync to mimic RwLock behavior.
        // This allows ArcUnsafe to be shared across threads, assuming the user logic handles race conditions.
        unsafe impl<T: Send + Sync> Sync for ArcUnsafe<T> {}
        unsafe impl<T: Send + Sync> Send for ArcUnsafe<T> {}
        unsafe impl<T: Send + Sync> Sync for WeakUnsafe<T> {}
        unsafe impl<T: Send + Sync> Send for WeakUnsafe<T> {}

        impl<T> ArcUnsafe<T> {
            pub fn downgrade(&self) -> WeakUnsafe<T> {
                WeakUnsafe::<T> {
                    ptr: Arc::downgrade(&self.ptr)
                }
            }
        }

        impl<T> WeakUnsafe<T> {
            pub fn upgrade_force(&self) -> ArcUnsafe<T> {
                ArcUnsafe::<T> {
                    ptr: self.ptr.upgrade().unwrap()
                }
            }
            pub fn upgrade(&self) -> Option<ArcUnsafe<T>> {
                self.ptr.upgrade().map(|x| ArcUnsafe::<T> { ptr: x })
            }
            pub fn ptr_eq(&self, other: &Self) -> bool {
                Weak::ptr_eq(&self.ptr, &other.ptr)
            }
        }

        impl<T> Clone for ArcUnsafe<T> {
            fn clone(&self) -> Self {
                Self { ptr: self.ptr.clone() }
            }
        }

        impl<T> UnsafePtr<T> for ArcUnsafe<T> {
            fn new_ptr(ptr: Arc<UnsafeCell<T>>) -> Self { Self { ptr }  }
            fn new_value(obj: T) -> Self { Self::new_ptr(Arc::new(UnsafeCell::new(obj))) }
            #[inline(always)] fn ptr(&self) -> &Arc<UnsafeCell<T>> { &self.ptr }
            #[inline(always)] fn ptr_mut(&mut self) -> &mut Arc<UnsafeCell<T>> { &mut self.ptr }
        }

        impl<T> WeakUnsafe<T> {
            #[inline(always)] pub fn ptr(&self) -> &Weak<UnsafeCell<T>> { &self.ptr }
            #[inline(always)] pub fn ptr_mut(&mut self) -> &mut Weak<UnsafeCell<T>> { &mut self.ptr }
        }

        impl<T> PartialEq for ArcUnsafe<T> {
            fn eq(&self, other: &Self) -> bool { self.ptr_eq(other) }
        }
        impl<T> Eq for ArcUnsafe<T> { }

        impl<T> Clone for WeakUnsafe<T> {
            fn clone(&self) -> Self {
                Self { ptr: self.ptr.clone() }
            }
        }

        impl<T> PartialEq for WeakUnsafe<T> {
            fn eq(&self, other: &Self) -> bool { self.ptr.ptr_eq(&other.ptr) }
        }
        impl<T> Eq for WeakUnsafe<T> { }

        // IDENTITY HASHING
        impl<T> std::hash::Hash for ArcUnsafe<T> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                let address = Arc::as_ptr(&self.ptr);
                address.hash(state);
            }
        }
        
        impl<T> std::hash::Hash for WeakUnsafe<T> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                let address = Weak::as_ptr(&self.ptr);
                address.hash(state);
            }
        }

        // WeakTable Integration
        impl<T> weak_table::traits::WeakElement for WeakUnsafe<T> {
            type Strong = ArcUnsafe<T>;

            fn new(view: &ArcUnsafe<T>) -> Self {
                view.downgrade()
            }

            fn view(&self) -> Option<ArcUnsafe<T>> {
                self.upgrade()
            }

            fn clone(view: &ArcUnsafe<T>) -> ArcUnsafe<T> {
                view.clone()
            }
        }

        // Deref (Exposes UnsafeCell)
        impl<T> std::ops::Deref for ArcUnsafe<T> {
            type Target = std::cell::UnsafeCell<T>;
            fn deref(&self) -> &Self::Target {
                &self.ptr
            }
        }

        // --- TYPE ALIASES ---
        pub type ArcManualSafeLock<T> = ArcUnsafe<T>;
        pub type WeakManualSafeLock<T> = WeakUnsafe<T>;
    }
}

// =========================================================================================
//  SAFE IMPLEMENTATION (DEFAULT)
//  Uses parking_lot::RwLock for standard thread safety.
// =========================================================================================
cfg_if::cfg_if! {
    if #[cfg(not(feature="unsafe_pointer"))] {
        use crate::parking_lot::lock_api::{RwLockReadGuard, RwLockWriteGuard};
        use crate::parking_lot::{RawRwLock, RwLock};

        pub trait RwLockPtr<ObjType> {
            fn new_ptr(ptr: Arc<RwLock<ObjType>>) -> Self;
            fn new_value(obj: ObjType) -> Self;
            fn ptr(&self) -> &Arc<RwLock<ObjType>>;
            fn ptr_mut(&mut self) -> &mut Arc<RwLock<ObjType>>;

            #[inline(always)]
            fn read_recursive(&self) -> RwLockReadGuard<RawRwLock, ObjType> {
                self.ptr().read_recursive()
            }

            #[inline(always)]
            fn write(&self) -> RwLockWriteGuard<RawRwLock, ObjType> {
                self.ptr().write()
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
            fn new_ptr(ptr: Arc<RwLock<T>>) -> Self { Self { ptr } }
            fn new_value(obj: T) -> Self { Self::new_ptr(Arc::new(RwLock::new(obj))) }
            #[inline(always)] fn ptr(&self) -> &Arc<RwLock<T>> { &self.ptr }
            #[inline(always)] fn ptr_mut(&mut self) -> &mut Arc<RwLock<T>> { &mut self.ptr }
        }

        impl<T: Send + Sync> WeakRwLock<T> {
            #[inline(always)] pub fn ptr(&self) -> &Weak<RwLock<T>> { &self.ptr }
            #[inline(always)] fn ptr_mut(&mut self) -> &mut Weak<RwLock<T>> { &mut self.ptr }
        }

        impl<T: Send + Sync> PartialEq for ArcRwLock<T> {
            fn eq(&self, other: &Self) -> bool { self.ptr_eq(other) }
        }
        impl<T: Send + Sync> Eq for ArcRwLock<T> {}

        impl<T> Clone for WeakRwLock<T> {
            fn clone(&self) -> Self { Self { ptr: self.ptr.clone() } }
        }

        impl<T> PartialEq for WeakRwLock<T> {
            fn eq(&self, other: &Self) -> bool { self.ptr.ptr_eq(&other.ptr) }
        }
        impl<T> Eq for WeakRwLock<T> {}

        // IDENTITY HASHING
        impl<T: Send + Sync> std::hash::Hash for ArcRwLock<T> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                let address = Arc::as_ptr(&self.ptr);
                address.hash(state);
            }
        }

        impl<T: Send + Sync> std::hash::Hash for WeakRwLock<T> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                let address = Weak::as_ptr(&self.ptr);
                address.hash(state);
            }
        }

        // WeakTable Integration
        impl<T: Send + Sync> weak_table::traits::WeakElement for WeakRwLock<T> {
            type Strong = ArcRwLock<T>;

            fn new(view: &ArcRwLock<T>) -> Self {
                view.downgrade()
            }

            fn view(&self) -> Option<ArcRwLock<T>> {
                self.upgrade()
            }

            fn clone(view: &ArcRwLock<T>) -> ArcRwLock<T> {
                view.clone()
            }
        }

        // Deref (Exposes RwLock)
        impl<T> std::ops::Deref for ArcRwLock<T> {
            type Target = RwLock<T>;
            fn deref(&self) -> &Self::Target {
                &self.ptr
            }
        }

        // --- TYPE ALIASES ---
        pub type ArcManualSafeLock<T> = ArcRwLock<T>;
        pub type WeakManualSafeLock<T> = WeakRwLock<T>;
    }
}

// =========================================================================================
//  TESTS
// =========================================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Tester {
        idx: usize,
    }

    // Dynamic Type Alias for Testing
    cfg_if::cfg_if! {
        if #[cfg(feature="unsafe_pointer")] {
            type TesterPtr = ArcUnsafe<Tester>;
            type TesterWeak = WeakUnsafe<Tester>;
            use UnsafePtr as PtrTrait; 
        } else {
            type TesterPtr = ArcRwLock<Tester>;
            type TesterWeak = WeakRwLock<Tester>;
            use RwLockPtr as PtrTrait; 
        }
    }

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

        // Testing Write
        {
            // Note: In unsafe mode, write() returns &mut T directly.
            // In safe mode, write() returns a Guard that derefs to mutable.
            // This syntax works for both due to Deref.
            ptr.write().idx = 1;
        }

        // Testing Weak Upgrade and Read
        assert_eq!(weak.upgrade_force().read_recursive().idx, 1);

        // Testing Write via Weak
        {
            weak.upgrade_force().write().idx = 2;
        }
        
        assert_eq!(ptr.read_recursive().idx, 2);
    }
}