use std::path::{Path, PathBuf};

fn main() {
    let kernels = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let metal = std::env::var_os("CARGO_FEATURE_METAL").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos";
    let cuda = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    assert!(
        !(metal && cuda),
        "det-sys: the metal and cuda features are exclusive"
    );
    if metal {
        build_metal(&kernels.join("metal"));
    } else if cuda {
        build_cuda(&kernels.join("cuda"));
    }
}

fn build_metal(kernels: &Path) {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shaders = kernels.join("shaders");
    let mut names: Vec<_> = std::fs::read_dir(&shaders)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".metal") && n != "common.metal")
        .collect();
    names.sort();
    let mut src = String::new();
    src.push_str(&std::fs::read_to_string(kernels.join("params.h")).unwrap());
    src.push('\n');
    src.push_str(&std::fs::read_to_string(shaders.join("common.metal")).unwrap());
    for n in &names {
        src.push_str(&format!("\n// ===== {n}\n"));
        src.push_str(&std::fs::read_to_string(shaders.join(n)).unwrap());
        src.push('\n');
    }
    assert!(
        !src.contains(")DETSRC\""),
        "shader source contains the raw string terminator"
    );
    let embed = out.join("shaders_embed.mm");
    std::fs::write(
        &embed,
        format!(
            "extern \"C\" const char* const det_embedded_shaders = R\"DETSRC({src})DETSRC\";\n"
        ),
    )
    .unwrap();

    let sources = [
        "det_metal.mm",
        "glue.mm",
        "gemm.mm",
        "attn.mm",
        "gdn.mm",
        "sample.mm",
    ];
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .include(kernels)
        .flag("-ObjC++")
        .flag("-fobjc-arc")
        .flag("-std=c++17")
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .flag("-Wno-unused-function")
        .opt_level(2);
    for s in sources {
        build.file(kernels.join(s));
    }
    build.file(&embed);
    build.compile("det_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    for s in sources
        .iter()
        .chain(["det_metal.h", "runtime.h", "params.h"].iter())
    {
        println!("cargo:rerun-if-changed={}", kernels.join(s).display());
    }
    println!("cargo:rerun-if-changed={}", shaders.display());
    for n in names
        .iter()
        .chain(std::iter::once(&"common.metal".to_string()))
    {
        println!("cargo:rerun-if-changed={}", shaders.join(n).display());
    }
}

/// nvcc from `CUDA_HOME` (or `CUDA_PATH`, or /usr/local/cuda); `DET_CUDA_ARCHS` lists the SM versions
/// to compile for (default "120,89,80"). Never fast math: `-fmad=false` keeps every fp32 op as written.
fn build_cuda(kernels: &Path) {
    let home = ["CUDA_HOME", "CUDA_PATH"]
        .iter()
        .find_map(std::env::var_os)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let archs = std::env::var("DET_CUDA_ARCHS").unwrap_or_else(|_| "120,89,80".into());
    let sources = [
        "runtime.cu",
        "norm.cu",
        "gemm.cu",
        "attn.cu",
        "gdn.cu",
        "sample.cu",
    ];
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .cudart("static")
        .compiler(home.join("bin/nvcc"))
        .include(kernels)
        .no_default_flags(true)
        .warnings(false);
    build
        .flag("-O3")
        .flag("-std=c++17")
        .flag("-fmad=false")
        .flag("-lineinfo")
        .flag("-Xcompiler")
        .flag("-fPIC,-ffp-contract=off");
    for a in archs.split(',') {
        build
            .flag("-gencode")
            .flag(format!("arch=compute_{a},code=sm_{a}"));
    }
    for s in sources {
        build.file(kernels.join(s));
    }
    build.compile("det_cuda");
    println!(
        "cargo:rustc-link-search=native={}",
        home.join("lib64").display()
    );
    for l in ["stdc++", "rt", "dl", "pthread"] {
        println!("cargo:rustc-link-lib={l}");
    }
    for s in sources.iter().chain(["det_cuda.h", "common.cuh"].iter()) {
        println!("cargo:rerun-if-changed={}", kernels.join(s).display());
    }
    for v in ["DET_CUDA_ARCHS", "CUDA_HOME", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
}
