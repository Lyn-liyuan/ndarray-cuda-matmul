fn main() {
    println!("cargo:rustc-env=PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig");
    println!("cargo:rustc-link-search={}", "/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search={}", "/disk/lyn/workspace/matmul");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cblas"); 
}