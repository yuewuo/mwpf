//! Pointer Types
//!
//! This module provides a unified interface for reference-counted objects with interior mutability.
//! 
//! Feature Flags:
//! - Default: Uses `OrdArcRwLock` (Safe, Thread-safe, Ordered).
//! - `unsafe_pointer`: Uses `OrdArcUnsafe` (Unsafe, High-performance, Ordered).

#![cfg_attr(feature = "unsafe_pointer", allow(dropping_references))]

use std::sync::{Arc, Weak};
use std::hash::Hash;

// =========================================================================================
//  UNSAFE IMPLEMENTATION (feature = "unsafe_pointer")
//  Uses UnsafeCell for maximum performance, bypassing locks.
//  Includes Standard (ArcUnsafe) and Ordered (OrdArcUnsafe) variants.
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

        // --- STRUCT DEFINITIONS ---

        // 1. Standard ArcUnsafe (Identity based)
        pub struct ArcUnsafe<T> {
            ptr: Arc<UnsafeCell<T>>,
        }

        pub struct WeakUnsafe<T> {
            ptr: Weak<UnsafeCell<T>>,
        }

        // 2. Ordered ArcUnsafe (Sort key based)
        pub struct OrdArcUnsafe<T, U: Ord + Copy + Default + Hash> {
            ord: U,
            ptr: Arc<UnsafeCell<T>>,
        }

        pub struct OrdWeakUnsafe<T, U: Ord + Copy + Default + Hash> {
            ord: U,
            ptr: Weak<UnsafeCell<T>>,
        }

        // --- MANUAL SEND/SYNC ---
        // SAFETY: We mimic RwLock behavior. The user is responsible for race conditions.
        unsafe impl<T: Send + Sync> Sync for ArcUnsafe<T> {}
        unsafe impl<T: Send + Sync> Send for ArcUnsafe<T> {}
        unsafe impl<T: Send + Sync> Sync for WeakUnsafe<T> {}
        unsafe impl<T: Send + Sync> Send for WeakUnsafe<T> {}

        unsafe impl<T: Send + Sync, U: Ord + Copy + Default + Hash + Send + Sync> Sync for OrdArcUnsafe<T, U> {}
        unsafe impl<T: Send + Sync, U: Ord + Copy + Default + Hash + Send + Sync> Send for OrdArcUnsafe<T, U> {}
        unsafe impl<T: Send + Sync, U: Ord + Copy + Default + Hash + Send + Sync> Sync for OrdWeakUnsafe<T, U> {}
        unsafe impl<T: Send + Sync, U: Ord + Copy + Default + Hash + Send + Sync> Send for OrdWeakUnsafe<T, U> {}

        // --- IMPLEMENTATIONS: ArcUnsafe (Standard) ---
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

        impl<T> weak_table::traits::WeakElement for WeakUnsafe<T> {
            type Strong = ArcUnsafe<T>;
            fn new(view: &ArcUnsafe<T>) -> Self { view.downgrade() }
            fn view(&self) -> Option<ArcUnsafe<T>> { self.upgrade() }
            fn clone(view: &ArcUnsafe<T>) -> ArcUnsafe<T> { view.clone() }
        }

        impl<T> std::ops::Deref for ArcUnsafe<T> {
            type Target = std::cell::UnsafeCell<T>;
            fn deref(&self) -> &Self::Target { &self.ptr }
        }

        // --- IMPLEMENTATIONS: OrdArcUnsafe (Ordered) ---
        impl<T, U: Ord + Copy + Default + Hash> OrdArcUnsafe<T, U> {
            pub fn downgrade(&self) -> OrdWeakUnsafe<T, U> {
                OrdWeakUnsafe::<T, U> {
                    ord: self.ord,
                    ptr: Arc::downgrade(&self.ptr),
                }
            }
        }

        impl<T, U: Ord + Copy + Default + Hash> OrdWeakUnsafe<T, U> {
            pub fn upgrade_force(&self) -> OrdArcUnsafe<T, U> {
                OrdArcUnsafe::<T, U> {
                    ord: self.ord,
                    ptr: self.ptr.upgrade().unwrap(),
                }
            }
            pub fn upgrade(&self) -> Option<OrdArcUnsafe<T, U>> {
                self.ptr.upgrade().map(|x| OrdArcUnsafe::<T, U> { ord: self.ord, ptr: x })
            }
            pub fn ptr_eq(&self, other: &Self) -> bool {
                Weak::ptr_eq(&self.ptr, &other.ptr)
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash + Default> Clone for OrdArcUnsafe<T, U> {
            fn clone(&self) -> Self {
                Self::new_ptr(Arc::clone(self.ptr()), self.ord)
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash + Default> OrdArcUnsafe<T, U> {
            pub fn new_ptr(ptr: Arc<UnsafeCell<T>>, ord: U) -> Self {
                Self { ord, ptr }
            }
            pub fn new_value(obj: T, ord: U) -> Self {
                Self::new_ptr(Arc::new(UnsafeCell::new(obj)), ord)
            }
            #[inline(always)]
            pub fn ptr(&self) -> &Arc<UnsafeCell<T>> {
                &self.ptr
            }
            #[inline(always)]
            pub fn ptr_mut(&mut self) -> &mut Arc<UnsafeCell<T>> {
                &mut self.ptr
            }
            
            // Helper methods to match UnsafePtr-like usage
            #[inline(always)]
            pub fn read_recursive(&self) -> &T {
                unsafe { &*self.ptr.get() }
            }
            #[inline(always)]
            pub fn write(&self) -> &mut T {
                unsafe { &mut *self.ptr.get() }
            }

            #[inline(always)]
            pub fn get_ord(&self) -> U {
                self.ord
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> OrdWeakUnsafe<T, U> {
            #[inline(always)]
            pub fn ptr(&self) -> &Weak<UnsafeCell<T>> {
                &self.ptr
            }
        }

        // Ordering and Hashing for OrdArcUnsafe
        impl<T, U: Ord + Copy + Default + Hash> PartialEq for OrdArcUnsafe<T, U> {
            fn eq(&self, other: &Self) -> bool { self.ord.eq(&other.ord) }
        }
        impl<T, U: Ord + Copy + Default + Hash> Eq for OrdArcUnsafe<T, U> {}

        impl<T, U: Ord + Copy + Default + Hash> Ord for OrdArcUnsafe<T, U> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.ord.cmp(&other.ord) }
        }

        impl<T, U: Ord + Copy + Default + Hash> PartialOrd for OrdArcUnsafe<T, U> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
        }

        impl<T, U: Ord + Copy + Default + Hash> std::hash::Hash for OrdArcUnsafe<T, U> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.ord.hash(state); }
        }

        // Ordering and Hashing for OrdWeakUnsafe
        impl<T, U: Ord + Copy + Default + Hash> PartialEq for OrdWeakUnsafe<T, U> {
            fn eq(&self, other: &Self) -> bool { self.ord.eq(&other.ord) }
        }
        impl<T, U: Ord + Copy + Default + Hash> Eq for OrdWeakUnsafe<T, U> {}

        impl<T, U: Ord + Copy + Default + Hash> Ord for OrdWeakUnsafe<T, U> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.ord.cmp(&other.ord) }
        }
        
        impl<T, U: Ord + Copy + Default + Hash> PartialOrd for OrdWeakUnsafe<T, U> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
        }

        impl<T, U: Ord + Copy + Default + Hash> std::hash::Hash for OrdWeakUnsafe<T, U> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.ord.hash(state); }
        }

        impl<T, U: Ord + Copy + Default + Hash> Clone for OrdWeakUnsafe<T, U> {
            fn clone(&self) -> Self {
                Self { ord: self.ord, ptr: self.ptr.clone() }
            }
        }

        // WeakTable Integration
        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> weak_table::traits::WeakElement for OrdWeakUnsafe<T, U> {
            type Strong = OrdArcUnsafe<T, U>;
            fn new(view: &Self::Strong) -> Self { view.downgrade() }
            fn view(&self) -> Option<Self::Strong> { self.upgrade() }
            fn clone(view: &Self::Strong) -> Self::Strong { view.clone() }
        }

        impl<T, U: Ord + Copy + Default + Hash> std::ops::Deref for OrdArcUnsafe<T, U> {
            type Target = std::cell::UnsafeCell<T>;
            fn deref(&self) -> &Self::Target { &self.ptr }
        }

        // --- TYPE ALIASES ---
        // Defaults to Ordered Unsafe Pointer
        pub type ArcManualSafeLock<T> = OrdArcUnsafe<T, (usize, usize)>;
        pub type WeakManualSafeLock<T> = OrdWeakUnsafe<T, (usize, usize)>;
    }
}

// =========================================================================================
//  SAFE IMPLEMENTATION (DEFAULT)
//  Uses parking_lot::RwLock for standard thread safety.
//  Includes Standard (ArcRwLock) and Ordered (OrdArcRwLock) variants.
// =========================================================================================
cfg_if::cfg_if! {
    if #[cfg(not(feature="unsafe_pointer"))] {
        use crate::parking_lot::lock_api::{RwLockReadGuard, RwLockWriteGuard};
        use crate::parking_lot::{RawRwLock, RwLock};

        // --- TRAIT DEFINITION ---
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

        // --- STRUCT DEFINITIONS ---

        // 1. Standard ArcRwLock
        pub struct ArcRwLock<T> {
            ptr: Arc<RwLock<T>>,
        }

        pub struct WeakRwLock<T> {
            ptr: Weak<RwLock<T>>,
        }

        // 2. Ordered ArcRwLock (For deterministic sorting)
        pub struct OrdArcRwLock<T, U: Ord + Copy + Default + Hash> {
            pub ord: U,
            pub ptr: Arc<RwLock<T>>,
        }

        pub struct OrdWeakRwLock<T, U: Ord + Copy + Default + Hash> {
            pub ord: U,
            pub ptr: Weak<RwLock<T>>,
        }

        // --- IMPLEMENTATIONS: ArcRwLock (Standard) ---
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

        impl<T: Send + Sync> PartialEq for WeakRwLock<T> {
            fn eq(&self, other: &Self) -> bool { self.ptr.ptr_eq(&other.ptr) }
        }
        impl<T: Send + Sync> Eq for WeakRwLock<T> {}

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

        impl<T: Send + Sync> weak_table::traits::WeakElement for WeakRwLock<T> {
            type Strong = ArcRwLock<T>;
            fn new(view: &ArcRwLock<T>) -> Self { view.downgrade() }
            fn view(&self) -> Option<ArcRwLock<T>> { self.upgrade() }
            fn clone(view: &ArcRwLock<T>) -> ArcRwLock<T> { view.clone() }
        }

        impl<T> std::ops::Deref for ArcRwLock<T> {
            type Target = RwLock<T>;
            fn deref(&self) -> &Self::Target { &self.ptr }
        }

        // --- IMPLEMENTATIONS: OrdArcRwLock (Ordered) ---
        impl<T, U: Ord + Copy + Default + Hash> OrdArcRwLock<T, U> {
            pub fn downgrade(&self) -> OrdWeakRwLock<T, U> {
                OrdWeakRwLock::<T, U> {
                    ord: self.ord,
                    ptr: Arc::downgrade(&self.ptr),
                }
            }
        }

        impl<T, U: Ord + Copy + Default + Hash> OrdWeakRwLock<T, U> {
            pub fn upgrade_force(&self) -> OrdArcRwLock<T, U> {
                OrdArcRwLock::<T, U> {
                    ord: self.ord,
                    ptr: self.ptr.upgrade().unwrap(),
                }
            }
            pub fn upgrade(&self) -> Option<OrdArcRwLock<T, U>> {
                self.ptr.upgrade().map(|x| OrdArcRwLock::<T, U> { ord: self.ord, ptr: x })
            }
            pub fn ptr_eq(&self, other: &Self) -> bool {
                Weak::ptr_eq(&self.ptr, &other.ptr)
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash + Default> Clone for OrdArcRwLock<T, U> {
            fn clone(&self) -> Self {
                Self::new_ptr(Arc::clone(self.ptr()), self.ord)
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash + Default> OrdArcRwLock<T, U> {
            pub fn new_ptr(ptr: Arc<RwLock<T>>, ord: U) -> Self {
                Self { ord, ptr }
            }
            pub fn new_value(obj: T, ord: U) -> Self {
                Self::new_ptr(Arc::new(RwLock::new(obj)), ord)
            }
            #[inline(always)]
            pub fn ptr(&self) -> &Arc<RwLock<T>> {
                &self.ptr
            }
            #[inline(always)]
            pub fn ptr_mut(&mut self) -> &mut Arc<RwLock<T>> {
                &mut self.ptr
            }
            #[inline(always)]
            pub fn read_recursive(&self) -> RwLockReadGuard<RawRwLock, T> {
                self.ptr.read_recursive()
            }
            #[inline(always)]
            pub fn write(&self) -> RwLockWriteGuard<RawRwLock, T> {
                self.ptr.write()
            }
            #[inline(always)]
            pub fn get_ord(&self) -> U {
                self.ord
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> OrdWeakRwLock<T, U> {
            #[inline(always)]
            pub fn ptr(&self) -> &Weak<RwLock<T>> {
                &self.ptr
            }
            #[inline(always)]
            fn ptr_mut(&mut self) -> &mut Weak<RwLock<T>> {
                &mut self.ptr
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> PartialEq for OrdArcRwLock<T, U> {
            fn eq(&self, other: &Self) -> bool {
                self.ord.eq(&other.ord)
            }
        }
        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> Eq for OrdArcRwLock<T, U> {}

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> Ord for OrdArcRwLock<T, U> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.ord.cmp(&other.ord)
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> Ord for OrdWeakRwLock<T, U> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.ord.cmp(&other.ord)
            }
        }
        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> Eq for OrdWeakRwLock<T, U> {}

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> PartialEq for OrdWeakRwLock<T, U> {
            fn eq(&self, other: &Self) -> bool {
                self.ord.eq(&other.ord)
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> PartialOrd for OrdArcRwLock<T, U> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> PartialOrd for OrdWeakRwLock<T, U> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> Clone for OrdWeakRwLock<T, U> {
            fn clone(&self) -> Self {
                Self {
                    ord: self.ord,
                    ptr: self.ptr.clone(),
                }
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> std::ops::Deref for OrdArcRwLock<T, U> {
            type Target = RwLock<T>;
            fn deref(&self) -> &Self::Target {
                &self.ptr
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> std::hash::Hash for OrdArcRwLock<T, U> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                let address = Arc::as_ptr(&self.ptr);
                (address, self.ord).hash(state);
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> std::hash::Hash for OrdWeakRwLock<T, U> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                let address = Weak::as_ptr(&self.ptr);
                (address, self.ord).hash(state);
            }
        }

        impl<T: Send + Sync, U: Ord + Copy + Default + Hash> weak_table::traits::WeakElement for OrdWeakRwLock<T, U> {
            type Strong = OrdArcRwLock<T, U>;
            fn new(view: &Self::Strong) -> Self { view.downgrade() }
            fn view(&self) -> Option<Self::Strong> { self.upgrade() }
            fn clone(view: &Self::Strong) -> Self::Strong { view.clone() }
        }

        // --- TYPE ALIASES ---
        // Defaults to Ordered Safe Lock with a default tuple key
        // pub type ArcManualSafeLock<T> = ArcRwLock<T>;
        // pub type WeakManualSafeLock<T> = WeakRwLock<T>;
        pub type ArcManualSafeLock<T> = OrdArcRwLock<T, (usize, usize)>;
        pub type WeakManualSafeLock<T> = OrdWeakRwLock<T, (usize, usize)>;
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
    // NOTE: Both now point to their respective Ordered variants for consistency.
    cfg_if::cfg_if! {
        if #[cfg(feature="unsafe_pointer")] {
            type TesterPtr = ArcManualSafeLock<Tester>;
            type TesterWeak = WeakManualSafeLock<Tester>;
        } else {
            type TesterPtr = ArcManualSafeLock<Tester>;
            type TesterWeak = WeakManualSafeLock<Tester>;
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
        let ptr = TesterPtr::new_value(Tester { idx: 0 }, (0, 0));
        let weak = ptr.downgrade();

        // Testing Write
        {
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