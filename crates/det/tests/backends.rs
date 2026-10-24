//! The conformance suites (`det::conformance`): every op of the GPU and x86 backends against the
//! reference backend, bit for bit.

use det::conformance::{self, Report};

fn check(r: Report) {
    let failures: Vec<String> = r.failures().map(|(n, m)| format!("{n}: {m}")).collect();
    assert!(
        failures.is_empty(),
        "{} of {} checks failed:\n{}",
        failures.len(),
        r.checks.len(),
        failures.join("\n")
    );
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
mod gpu {
    use super::*;
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

    #[test]
    fn primitives() {
        let _g = serial();
        let mut r = Report::default();
        conformance::primitives(&test_prim, &mut r);
        check(r);
    }

    macro_rules! suites {
        ($($name:ident),*) => {$(
            #[test]
            fn $name() {
                let _g = serial();
                let mut r = Report::default();
                conformance::$name(&Gpu::new(), &mut r);
                check(r);
            }
        )*};
    }
    suites!(gemm, norms, attention, gdn, sampling);
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use det::x86::X86;

    /// Both dot-product paths where the CPU has VNNI, the AVX2 one otherwise.
    fn each(f: fn(&X86, &mut Report)) {
        if !X86::available() {
            return;
        }
        let mut r = Report::default();
        f(&X86::with_vnni(false), &mut r);
        if is_x86_feature_detected!("avxvnni") {
            f(&X86::with_vnni(true), &mut r);
        }
        check(r);
    }

    macro_rules! suites {
        ($($name:ident),*) => {$(
            #[test]
            fn $name() {
                each(conformance::$name::<X86>);
            }
        )*};
    }
    suites!(gemm, norms, attention, gdn, sampling);
}
