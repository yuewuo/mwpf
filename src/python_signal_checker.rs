//! Python signals need manual checking, otherwise we cannot Ctrl-C to stop the program.
//! It is better to allow the user to stop it early if a decoding task takes too long, say tens of seconds,
//! rather than having to kill the process in the task manager (often killing the whole terminal)
//!
//! The signal can only be checked in the main thread (see https://pyo3.rs/main/doc/pyo3/marker/struct.python#method.check_signals),
//! but we need to do computation in the main thread, so we cannot afford to get the GIL frequently to query the signal.
//! Here we use another thread to set an atomic bool every 100ms.
//! The main loop will only get GIL and query for interrupt signal when this atomic bool is asserted, and then clear it.
//! To avoid this behavior affecting some very short tasks, we set another atomic flag called `skip_next`.
//! If the ticker sees the `skip_next` flag, it will clear it and do not set the `should_check` flag.

use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{sleep, spawn, JoinHandle};
use std::time::Duration;
use thread_priority::{set_current_thread_priority, ThreadPriority};

pub fn force_check_signals() -> PyResult<()> {
    Python::with_gil(|py| Python::check_signals(py))
}

pub struct PythonSignalChecker {
    pub should_check: &'static AtomicBool,
    pub is_skipping_next: &'static AtomicBool,
    pub thread_handler: JoinHandle<()>,
}

pub static DEFAULT_SHOULD_CHECK: AtomicBool = AtomicBool::new(false);
pub static DEFAULT_IS_SKIPPING_NEXT: AtomicBool = AtomicBool::new(false);
pub static DEFAULT_SLEEP_INTERVAL_MILLIS: u64 = 100;
lazy_static! {
    pub static ref PYTHON_SIGNAL_CHECKER: PythonSignalChecker = PythonSignalChecker::new(
        &DEFAULT_SHOULD_CHECK,
        &DEFAULT_IS_SKIPPING_NEXT,
        DEFAULT_SLEEP_INTERVAL_MILLIS
    );
}

impl PythonSignalChecker {
    pub fn new(
        should_check: &'static AtomicBool,
        is_skipping_next: &'static AtomicBool,
        sleep_interval_millis: u64,
    ) -> Self {
        let thread_handler = spawn(move || {
            let _ = set_current_thread_priority(ThreadPriority::Min);
            loop {
                if is_skipping_next.swap(false, Ordering::Relaxed) {
                    continue;
                }
                should_check.store(true, Ordering::Relaxed);
                sleep(Duration::from_millis(sleep_interval_millis));
            }
        });
        Self {
            should_check,
            is_skipping_next,
            thread_handler,
        }
    }

    #[inline]
    pub fn check(&self) -> PyResult<()> {
        if self.should_check.swap(false, Ordering::Relaxed) {
            force_check_signals()?;
        }
        Ok(())
    }

    #[inline]
    pub fn skip_next(&self) {
        self.is_skipping_next.store(true, Ordering::Relaxed);
        self.should_check.store(false, Ordering::Relaxed);
    }
}
