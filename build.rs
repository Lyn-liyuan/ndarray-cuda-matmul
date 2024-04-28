
fn main() {
    let builder = bindgen_cuda::Builder::default();
    builder.build_lib("libcuda.a");
    println!("cargo:rustc-link-search={}", "/mnt/lyn/workspace/matmul");
    println!("cargo:rustc-link-search={}", "/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=static=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cblas");
}
