//! The conformance suites (`det::conformance`): every GPU op against the CPU backend, bit for bit.

#![cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]

use det::conformance::{self, Report};
#[cfg(feature = "cuda")]
use det::cuda::{test_prim, Cuda as Gpu};
#[cfg(not(feature = "cuda"))]
use det::metal::{test_prim, Metal as Gpu};
use std::sync::{Mutex, MutexGuard};

/// The GPU runtime is one global in-order queue: the tests take turns.
static LOCK: Mutex<()> = Mutex::new(());
fn serial() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn check(r: Report) {
    let failures: Vec<String> = r.failures().map(|(n, m)| format!("{n}: {m}")).collect();
    assert!(failures.is_empty(), "{} of {} checks failed:\n{}", failures.len(), r.checks.len(), failures.join("\n"));
}

#[test]
fn primitives() {
    let _g = serial();
    let mut r = Report::default();
    conformance::primitives(&test_prim, &mut r);
    check(r);
}

#[test]
fn gemm() {
    let _g = serial();
    let mut r = Report::default();
    conformance::gemm(&Gpu::new(), &mut r);
    check(r);
}

#[test]
fn norms() {
    let _g = serial();
    let mut r = Report::default();
    conformance::norms(&Gpu::new(), &mut r);
    check(r);
}

#[test]
fn attention() {
    let _g = serial();
    let mut r = Report::default();
    conformance::attention(&Gpu::new(), &mut r);
    check(r);
}

#[test]
fn gdn() {
    let _g = serial();
    let mut r = Report::default();
    conformance::gdn(&Gpu::new(), &mut r);
    check(r);
}

#[test]
fn sampling() {
    let _g = serial();
    let mut r = Report::default();
    conformance::sampling(&Gpu::new(), &mut r);
    check(r);
}
