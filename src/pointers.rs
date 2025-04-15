//! Pointer Types
//!
#![cfg_attr(feature = "unsafe_pointer", allow(dropping_references))]

use crate::parking_lot::lock_api::{RwLockReadGuard, RwLockWriteGuard};
use crate::parking_lot::{RawRwLock, RwLock};
use std::collections::BTreeSet;
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

impl<T: Send + Sync> WeakRwLock<T> {
    #[inline(always)]
    pub fn ptr(&self) -> &Weak<RwLock<T>> {
        &self.ptr
    }
    #[inline(always)]
    fn ptr_mut(&mut self) -> &mut Weak<RwLock<T>> {
        &mut self.ptr
    }
}

impl<T: Send + Sync> PartialEq for ArcRwLock<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
    }
}

impl<T: Send + Sync> Eq for ArcRwLock<T> {}

impl<T: Send + Sync> Ord for ArcRwLock<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let ptr1 = Arc::as_ptr(self.ptr());
        let ptr2 = Arc::as_ptr(other.ptr());
        ptr1.cmp(&ptr2)
    }
}

impl<T: Send + Sync> Ord for WeakRwLock<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let ptr1 = Weak::as_ptr(self.ptr());
        let ptr2 = Weak::as_ptr(other.ptr());
        ptr1.cmp(&ptr2)
    }
}

impl<T: Send + Sync> PartialOrd for ArcRwLock<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Send + Sync> PartialOrd for WeakRwLock<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

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

impl<T: Send + Sync> std::hash::Hash for ArcRwLock<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let address = Arc::as_ptr(&self.ptr);
        address.hash(state);
        // self.ptr.hash(state);

        //    std::ptr::hash(self.ptr(), state);
    }
}

impl<T: Send + Sync> std::hash::Hash for WeakRwLock<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let address = Weak::as_ptr(&self.ptr);
        address.hash(state);
        //    std::ptr::hash(self, state);
    }
}

impl<T: Send + Sync> weak_table::traits::WeakElement for WeakRwLock<T> {
    type Strong = ArcRwLock<T>;
    fn new(view: &Self::Strong) -> Self {
        view.downgrade()
    }
    fn view(&self) -> Option<Self::Strong> {
        self.upgrade()
    }
    fn clone(view: &Self::Strong) -> Self::Strong {
        view.clone()
    }
}

/*
 * unsafe APIs, used when maximizing speed performance
*/

cfg_if::cfg_if! {
    if #[cfg(feature="unsafe_pointer")] {

        // pub trait FastClearUnsafePtr<ObjType> where ObjType: FastClear {

        //     fn new_ptr(ptr: Arc<ObjType>) -> Self;

        //     fn new_value(obj: ObjType) -> Self;

        //     fn ptr(&self) -> &Arc<ObjType>;

        //     fn ptr_mut(&mut self) -> &mut Arc<ObjType>;

        //     #[inline(always)]
        //     fn read_recursive(&self, active_timestamp: FastClearTimestamp) -> &ObjType {
        //         let ret = self.ptr();
        //         ret.debug_assert_dynamic_cleared(active_timestamp);  // only assert during debug modes
        //         ret
        //     }

        //     /// without sanity check: this data might be outdated, so only use when you're read those immutable fields
        //     #[inline(always)]
        //     fn read_recursive_force(&self) -> &ObjType {
        //         self.ptr()
        //     }

        //     #[inline(always)]
        //     fn write(&self, active_timestamp: FastClearTimestamp) -> &mut ObjType {
        //         unsafe {
        //             // https://stackoverflow.com/questions/54237610/is-there-a-way-to-make-an-immutable-reference-mutable
        //             let ptr = self.ptr();
        //             let const_ptr = ptr as *const Arc<ObjType>;
        //             let mut_ptr = const_ptr as *mut Arc<ObjType>;
        //             let ret = Arc::get_mut_unchecked(&mut *mut_ptr);
        //             ret.debug_assert_dynamic_cleared(active_timestamp);  // only assert during debug modes
        //             ret
        //         }
        //     }

        //     #[inline(always)]
        //     fn try_write(&self, active_timestamp: FastClearTimestamp) -> Option<&mut ObjType> {
        //         Some(self.write(active_timestamp))
        //     }

        //     /// without sanity check: useful only in implementing hard_clear
        //     #[inline(always)]
        //     fn write_force(&self) -> &mut ObjType {
        //         unsafe {
        //             // https://stackoverflow.com/questions/54237610/is-there-a-way-to-make-an-immutable-reference-mutable
        //             let ptr = self.ptr();
        //             let const_ptr = ptr as *const Arc<ObjType>;
        //             let mut_ptr = const_ptr as *mut Arc<ObjType>;
        //             Arc::get_mut_unchecked(&mut *mut_ptr)
        //         }
        //     }

        //     /// dynamically clear it if not already cleared; it's safe to call many times, but it will acquire a writer lock
        //     #[inline(always)]
        //     fn dynamic_clear(&self, active_timestamp: FastClearTimestamp) {
        //         let value = self.write_force();
        //         value.dynamic_clear(active_timestamp);
        //     }

        //     fn ptr_eq(&self, other: &Self) -> bool {
        //         Arc::ptr_eq(self.ptr(), other.ptr())
        //     }

        // }

        pub trait UnsafePtr<ObjType> {

            fn new_ptr(ptr: Arc<ObjType>) -> Self;

            fn new_value(obj: ObjType) -> Self;

            fn ptr(&self) -> &Arc<ObjType>;

            fn ptr_mut(&mut self) -> &mut Arc<ObjType>;

            #[inline(always)]
            fn read_recursive(&self) -> &ObjType {
                self.ptr()
            }

            #[inline(always)]
            #[allow(invalid_reference_casting)]
            fn write(&self) -> &mut ObjType {
                unsafe {
                    // https://stackoverflow.com/questions/54237610/is-there-a-way-to-make-an-immutable-reference-mutable
                    let ptr = self.ptr();
                    let const_ptr = ptr as *const Arc<ObjType>;
                    let mut_ptr = const_ptr as *mut Arc<ObjType>;
                    Arc::get_mut_unchecked(&mut *mut_ptr)
                }
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
            ptr: Arc<T>,
        }

        pub struct WeakUnsafe<T> {
            ptr: Weak<T>,
        }

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
        }

        impl<T> Clone for ArcUnsafe<T> {
            fn clone(&self) -> Self {
                Self::new_ptr(Arc::clone(self.ptr()))
            }
        }

        impl<T> UnsafePtr<T> for ArcUnsafe<T> {
            fn new_ptr(ptr: Arc<T>) -> Self { Self { ptr }  }
            fn new_value(obj: T) -> Self { Self::new_ptr(Arc::new(obj)) }
            #[inline(always)] fn ptr(&self) -> &Arc<T> { &self.ptr }
            #[inline(always)] fn ptr_mut(&mut self) -> &mut Arc<T> { &mut self.ptr }
        }

        impl<T> WeakUnsafe<T> {
            #[inline(always)] pub fn ptr(&self) -> &Weak<T> { &self.ptr }
            #[inline(always)] pub fn ptr_mut(&mut self) -> &mut Weak<T> { &mut self.ptr }
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

        impl<T> std::ops::Deref for ArcUnsafe<T> {
            type Target = T;
            fn deref(&self) -> &Self::Target {
                &self.ptr
            }
        }

        impl<T> weak_table::traits::WeakElement for WeakUnsafe<T> {
            type Strong = ArcUnsafe<T>;
            fn new(view: &Self::Strong) -> Self {
                view.downgrade()
            }
            fn view(&self) -> Option<Self::Strong> {
                self.upgrade()
            }
            fn clone(view: &Self::Strong) -> Self::Strong {
                view.clone()
            }
        }

        impl<T: Send + Sync> std::hash::Hash for ArcUnsafe<T> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                let address = Arc::as_ptr(&self.ptr);
                address.hash(state);
                // self.ptr.hash(state);

            //    std::ptr::hash(self.ptr(), state);
            }
        }

        impl<T: Send + Sync> std::hash::Hash for WeakUnsafe<T> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                let address = Weak::as_ptr(&self.ptr);
                address.hash(state);
            //    std::ptr::hash(self, state);
            }
        }

        impl<T: Send + Sync> Ord for ArcUnsafe<T> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                let ptr1 = Arc::as_ptr(self.ptr());
                let ptr2 = Arc::as_ptr(other.ptr());
                ptr1.cmp(&ptr2)
            }
        }

        impl<T: Send + Sync> Ord for WeakUnsafe<T> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                let ptr1 = Weak::as_ptr(self.ptr());
                let ptr2 = Weak::as_ptr(other.ptr());
                ptr1.cmp(&ptr2)
            }
        }

        impl<T: Send + Sync> PartialOrd for ArcUnsafe<T> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl<T: Send + Sync> PartialOrd for WeakUnsafe<T> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        // impl<T> std::hash::Hash for ArcUnsafe<T> {
        //     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        //        std::ptr::hash(self, state);
        //     }
        // }

        // impl<T> std::hash::Hash for WeakUnsafe<T> {
        //     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        //        std::ptr::hash(self, state);
        //     }
        // }
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature="unsafe_pointer")] {
        // pub type FastClearArcManualSafeLock<T> = FastClearArcUnsafe<T>;
        // pub type FastClearWeakManualSafeLock<T> = FastClearWeakUnsafe<T>;
        pub type ArcManualSafeLock<T> = ArcUnsafe<T>;
        pub type WeakManualSafeLock<T> = WeakUnsafe<T>;

        // cfg_if::cfg_if! {
        //     if #[cfg(feature="dangerous_pointer")] {
        //         pub type FastClearArcManualSafeLockDangerous<T> = FastClearArcUnsafeDangerous<T>;
        //         pub type FastClearWeakManualSafeLockDangerous<T> = FastClearWeakUnsafeDangerous<T>;
        //     } else {
        //         // pub type FastClearArcManualSafeLockDangerous<T> = FastClearArcUnsafe<T>;
        //         // pub type FastClearWeakManualSafeLockDangerous<T> = FastClearWeakUnsafe<T>;
        //     }
        // }
    } else {
        pub type ArcManualSafeLock<T> = ArcRwLock<T>;
        pub type WeakManualSafeLock<T> = WeakRwLock<T>;
        // pub type FastClearArcManualSafeLockDangerous<T> = FastClearArcRwLock<T>;
        // pub type FastClearWeakManualSafeLockDangerous<T> = FastClearWeakRwLock<T>;
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

// //! Pointer Types
// //!
// #![cfg_attr(feature="unsafe_pointer", allow(dropping_references))]

// use crate::parking_lot::lock_api::{RwLockReadGuard, RwLockWriteGuard};
// use crate::parking_lot::{RawRwLock, RwLock};
// use std::sync::{Arc, Weak};
// use std::cell::UnsafeCell;

// pub trait RwLockPtr<ObjType> {
//     fn new_ptr(ptr: Arc<RwLock<ObjType>>) -> Self;

//     fn new_value(obj: ObjType) -> Self;

//     fn ptr(&self) -> &Arc<RwLock<ObjType>>;

//     fn ptr_mut(&mut self) -> &mut Arc<RwLock<ObjType>>;

//     #[inline(always)]
//     fn read_recursive(&self) -> RwLockReadGuard<RawRwLock, ObjType> {
//         let ret = self.ptr().read_recursive();
//         ret
//     }

//     #[inline(always)]
//     fn write(&self) -> RwLockWriteGuard<RawRwLock, ObjType> {
//         let ret = self.ptr().write();
//         ret
//     }

//     fn ptr_eq(&self, other: &Self) -> bool {
//         Arc::ptr_eq(self.ptr(), other.ptr())
//     }
// }

// pub struct ArcRwLock<T> {
//     ptr: Arc<RwLock<T>>,
// }

// pub struct WeakRwLock<T> {
//     ptr: Weak<RwLock<T>>,
// }

// impl<T> ArcRwLock<T> {
//     pub fn downgrade(&self) -> WeakRwLock<T> {
//         WeakRwLock::<T> {
//             ptr: Arc::downgrade(&self.ptr),
//         }
//     }
// }

// impl<T> WeakRwLock<T> {
//     pub fn upgrade_force(&self) -> ArcRwLock<T> {
//         ArcRwLock::<T> {
//             ptr: self.ptr.upgrade().unwrap(),
//         }
//     }
//     pub fn upgrade(&self) -> Option<ArcRwLock<T>> {
//         self.ptr.upgrade().map(|x| ArcRwLock::<T> { ptr: x })
//     }
//     pub fn ptr_eq(&self, other: &Self) -> bool {
//         Weak::ptr_eq(&self.ptr, &other.ptr)
//     }
// }

// impl<T: Send + Sync> Clone for ArcRwLock<T> {
//     fn clone(&self) -> Self {
//         Self::new_ptr(Arc::clone(self.ptr()))
//     }
// }

// impl<T: Send + Sync> RwLockPtr<T> for ArcRwLock<T> {
//     fn new_ptr(ptr: Arc<RwLock<T>>) -> Self {
//         Self { ptr }
//     }
//     fn new_value(obj: T) -> Self {
//         Self::new_ptr(Arc::new(RwLock::new(obj)))
//     }
//     #[inline(always)]
//     fn ptr(&self) -> &Arc<RwLock<T>> {
//         &self.ptr
//     }
//     #[inline(always)]
//     fn ptr_mut(&mut self) -> &mut Arc<RwLock<T>> {
//         &mut self.ptr
//     }
// }

// impl<T: Send + Sync> WeakRwLock<T> {
//     #[inline(always)]
//     pub fn ptr(&self) -> &Weak<RwLock<T>> {
//         &self.ptr
//     }
//     #[inline(always)]
//     pub fn ptr_mut(&mut self) -> &mut Weak<RwLock<T>> {
//         &mut self.ptr
//     }
// }

// impl<T: Send + Sync> PartialEq for ArcRwLock<T> {
//     fn eq(&self, other: &Self) -> bool {
//         self.ptr_eq(other)
//     }
// }

// impl<T: Send + Sync> Eq for ArcRwLock<T> {}

// impl<T: Send + Sync> Ord for ArcRwLock<T> {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         let ptr1 = Arc::as_ptr(self.ptr());
//         let ptr2 = Arc::as_ptr(other.ptr());
//         ptr1.cmp(&ptr2)
//     }
// }

// impl<T: Send + Sync> Ord for WeakRwLock<T> {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         let ptr1 = Weak::as_ptr(self.ptr());
//         let ptr2 = Weak::as_ptr(other.ptr());
//         ptr1.cmp(&ptr2)
//     }
// }

// impl<T: Send + Sync> PartialOrd for ArcRwLock<T> {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         Some(self.cmp(other))
//     }
// }

// impl<T: Send + Sync> PartialOrd for WeakRwLock<T> {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         Some(self.cmp(other))
//     }
// }

// impl<T> Clone for WeakRwLock<T> {
//     fn clone(&self) -> Self {
//         Self { ptr: self.ptr.clone() }
//     }
// }

// impl<T> PartialEq for WeakRwLock<T> {
//     fn eq(&self, other: &Self) -> bool {
//         self.ptr.ptr_eq(&other.ptr)
//     }
// }

// impl<T> Eq for WeakRwLock<T> {}

// impl<T> std::ops::Deref for ArcRwLock<T> {
//     type Target = RwLock<T>;
//     fn deref(&self) -> &Self::Target {
//         &self.ptr
//     }
// }

// impl<T: Send + Sync> std::hash::Hash for ArcRwLock<T> {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         let address = Arc::as_ptr(&self.ptr);
//         address.hash(state);
//     }
// }

// impl<T: Send + Sync> std::hash::Hash for WeakRwLock<T> {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         let address = Weak::as_ptr(&self.ptr);
//         address.hash(state);
//     }
// }

// /*
//     unsafe APIs, used when maximizing speed performance
// */
// cfg_if::cfg_if! {
//     if #[cfg(feature="unsafe_pointer")] {
//         pub trait UnsafePtr<ObjType> {
//             fn new_ptr(ptr: Arc<ObjType>) -> Self;

//             fn new_value(obj: ObjType) -> Self;

//             fn ptr(&self) -> &Arc<ObjType>;

//             fn ptr_mut(&mut self) -> &mut Arc<ObjType>;

//             #[inline(always)]
//             fn read_recursive(&self) -> &ObjType {
//                 self.ptr()
//             }

//             #[inline(always)]
//             fn write(&self) -> &mut ObjType {
//                 unsafe {
//                     // let ptr = self.ptr();
//                     // let cell: &UnsafeCell<ObjType> = &*ptr; // Assume `Arc<UnsafeCell<ObjType>>`
//                     // &mut *cell.get() // Safely access mutable reference

//                     // https://stackoverflow.com/questions/54237610/is-there-a-way-to-make-an-immutable-reference-mutable
//                     let ptr = self.ptr();
//                     let const_ptr = ptr as *const Arc<ObjType>;
//                     let mut_ptr = const_ptr as *mut Arc<ObjType>;
//                     Arc::get_mut_unchecked(&mut *mut_ptr)
//                 }
//             }

//             #[inline(always)]
//             fn try_write(&self) -> Option<&mut ObjType> {
//                 Some(self.write())
//             }

//             fn ptr_eq(&self, other: &Self) -> bool {
//                 Arc::ptr_eq(self.ptr(), other.ptr())
//             }
//         }

//         pub struct ArcUnsafe<T> {
//             ptr: Arc<T>,
//         }

//         pub struct WeakUnsafe<T> {
//             ptr: Weak<T>,
//         }

//         impl<T> ArcUnsafe<T> {
//             pub fn downgrade(&self) -> WeakUnsafe<T> {
//                 WeakUnsafe::<T> {
//                     ptr: Arc::downgrade(&self.ptr)
//                 }
//             }
//         }

//         impl<T> WeakUnsafe<T> {
//             pub fn upgrade_force(&self) -> ArcUnsafe<T> {
//                 ArcUnsafe::<T> {
//                     ptr: self.ptr.upgrade().unwrap()
//                 }
//             }
//             pub fn upgrade(&self) -> Option<ArcUnsafe<T>> {
//                 self.ptr.upgrade().map(|x| ArcUnsafe::<T> { ptr: x })
//             }
//         }

//         impl<T> Clone for ArcUnsafe<T> {
//             fn clone(&self) -> Self {
//                 Self::new_ptr(Arc::clone(self.ptr()))
//             }
//         }

//         impl<T> UnsafePtr<T> for ArcUnsafe<T> {
//             fn new_ptr(ptr: Arc<T>) -> Self { Self { ptr }  }
//             fn new_value(obj: T) -> Self { Self::new_ptr(Arc::new(obj)) }
//             #[inline(always)] fn ptr(&self) -> &Arc<T> { &self.ptr }
//             #[inline(always)] fn ptr_mut(&mut self) -> &mut Arc<T> { &mut self.ptr }
//         }

//         impl<T> WeakUnsafe<T> {
//             #[inline(always)] pub fn ptr(&self) -> &Weak<T> { &self.ptr }
//             #[inline(always)] pub fn ptr_mut(&mut self) -> &mut Weak<T> { &mut self.ptr }
//         }

//         impl<T> PartialEq for ArcUnsafe<T> {
//             fn eq(&self, other: &Self) -> bool { self.ptr_eq(other) }
//         }

//         impl<T> Eq for ArcUnsafe<T> { }

//         impl<T> Clone for WeakUnsafe<T> {
//             fn clone(&self) -> Self {
//                 Self { ptr: self.ptr.clone() }
//             }
//         }

//         impl<T> PartialEq for WeakUnsafe<T> {
//             fn eq(&self, other: &Self) -> bool { self.ptr.ptr_eq(&other.ptr) }
//         }

//         impl<T> Eq for WeakUnsafe<T> { }

//         impl<T> std::ops::Deref for ArcUnsafe<T> {
//             type Target = T;
//             fn deref(&self) -> &Self::Target {
//                 &self.ptr
//             }
//         }

//         impl<T> weak_table::traits::WeakElement for WeakUnsafe<T> {
//             type Strong = ArcUnsafe<T>;
//             fn new(view: &Self::Strong) -> Self {
//                 view.downgrade()
//             }
//             fn view(&self) -> Option<Self::Strong> {
//                 self.upgrade()
//             }
//             fn clone(view: &Self::Strong) -> Self::Strong {
//                 view.clone()
//             }
//         }

//         impl<T: Send + Sync> std::hash::Hash for ArcUnsafe<T> {
//             fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//                 let address = Arc::as_ptr(&self.ptr);
//                 address.hash(state);
//             }
//         }

//         impl<T: Send + Sync> std::hash::Hash for WeakUnsafe<T> {
//             fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//                 let address = Weak::as_ptr(&self.ptr);
//                 address.hash(state);
//             }
//         }

//         impl<T: Send + Sync> Ord for ArcUnsafe<T> {
//             fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//                 let ptr1 = Arc::as_ptr(self.ptr());
//                 let ptr2 = Arc::as_ptr(other.ptr());
//                 ptr1.cmp(&ptr2)
//             }
//         }

//         impl<T: Send + Sync> Ord for WeakUnsafe<T> {
//             fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//                 let ptr1 = Weak::as_ptr(self.ptr());
//                 let ptr2 = Weak::as_ptr(other.ptr());
//                 ptr1.cmp(&ptr2)
//             }
//         }

//         impl<T: Send + Sync> PartialOrd for ArcUnsafe<T> {
//             fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//                 Some(self.cmp(other))
//             }
//         }

//         impl<T: Send + Sync> PartialOrd for WeakUnsafe<T> {
//             fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//                 Some(self.cmp(other))
//             }
//         }
//     }
// }

// cfg_if::cfg_if! {
//     if #[cfg(feature="unsafe_pointer")] {
//         pub type ArcManualSafeLock<T> = ArcUnsafe<T>;
//         pub type WeakManualSafeLock<T> = WeakUnsafe<T>;
//     } else {
//         pub type ArcManualSafeLock<T> = ArcRwLock<T>;
//         pub type WeakManualSafeLock<T> = WeakRwLock<T>;
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[derive(Debug)]
//     struct Tester {
//         idx: usize,
//     }

//     type TesterPtr = ArcRwLock<Tester>;
//     type TesterWeak = WeakRwLock<Tester>;

//     impl std::fmt::Debug for TesterPtr {
//         fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//             let value = self.read_recursive();
//             write!(f, "{:?}", value)
//         }
//     }

//     impl std::fmt::Debug for TesterWeak {
//         fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//             self.upgrade_force().fmt(f)
//         }
//     }

//     #[test]
//     fn pointers_test_1() {
//         // cargo test pointers_test_1 -- --nocapture
//         let ptr = TesterPtr::new_value(Tester { idx: 0 });
//         let weak = ptr.downgrade();
//         ptr.write().idx = 1;
//         assert_eq!(weak.upgrade_force().read_recursive().idx, 1);
//         weak.upgrade_force().write().idx = 2;
//         assert_eq!(ptr.read_recursive().idx, 2);
//     }
// }
