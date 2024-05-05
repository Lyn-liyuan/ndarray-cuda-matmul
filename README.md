# Ndarray CUDA Matrix Multiplication

Welcome to the Ndarray CUDA Matrix Multiplication library, a high-performance computing solution designed to accelerate matrix operations using Nvidia's CUDA technology with Rust's ndarray data structure. This library leverages the powerful cuBLAS library to perform efficient matrix multiplications on compatible Nvidia GPUs.

## Features
- Seamless integration with Rust's ndarray crate.
- High-performance matrix multiplication utilizing CUDA.
- Support for one-dimensional and two-dimensional arrays.
- Automatic memory management between host and device.
- Simple and intuitive API mirroring that of ndarray.

## Prerequisites

To use this library, you will need:
- g++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
- rustc 1.77.2 (25ef9e3d8 2024-04-09)
- cuda_12.1
- Nvidia Driver Version: 535.154.05

## Usage

First, ensure that you have initialized the CUDA environment by calling init_cublas() before any matrix operations, and call destory_cublas() to clean up resources upon completion:

```Rust
extern crate ndarray_cuda_matmul;

use ndarray_cuda_matmul::{init_cublas, destory_cublas};

fn main() {
    // Initialize cublas context
    init_cublas();

    // Your matrix operations here

    // Clean up cublas context
    destory_cublas();
}
```

To perform matrix multiplication, use the cuda_dot method provided by the trait CudaDot implemented for ndarray’s ArrayBase:

```Rust
use ndarray::Array;
use ndarray_cuda_matmul::CudaDot;

let a = Array::from_shape_vec((m, k), vec![...]).unwrap();
let b = Array::from_shape_vec((k, n), vec![...]).unwrap();

let result = a.cuda_dot(&b);

```
Here m, n, and k represent the dimensions of the matrices, and vec![...] should be replaced with your actual data.

Using the method of first copying the matrix into GPU memory, here's a code example

```Rust
let a = array![[1.0_f32, 2.0_f32, 3.0_f32], [4.0_f32, 5.0_f32, 6.0_f32]];
let b = array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32], [5.0_f32, 6.0_f32]];
let c = array![[1.0f32,1.0f32],[1.0f32,1.0f32]];
init_cublas();
let out = a.to_device().dot(&b.to_device()).dot(&c.to_device()).to_host();
destory_cublas();
```
Matrix-scalar multiplication code example:

```Rust
    init_cublas();
    let out = a
        .to_device()
        .dot(&b.to_device())
        .dot(&c.to_device())
        .mul_scalar(2.0_f32)
        .to_host();
    destory_cublas();
```

Matrix inversion code example:

```Rust
let a = array![[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
                [2.0_f32, 3.0_f32, 1.0_f32, 2.0_f32],
                [1.0_f32, 1.0_f32, 1.0_f32, -1.0_f32],
                [1.0_f32, 0.0_f32, -2.0_f32, -6.0_f32],
            ];
    
init_cublas();
let out = a.to_device().inv().to_host();
destory_cublas();
```
Using run macro can simplify the code and write it like a mathematical expression. The following is an example of using run macro.

```Rust
fn least_squares_method()
{
    let x = array![[1f32, 1f32], [1f32, 2f32], [1f32, 3f32], [1f32, 4f32]];
    let y = array![[6f32], [5f32], [7f32], [10f32]];
    let bate_hat = run!(x,y => {
        let x_t = x.t();
        x_t.dot(x).inv().dot(&x_t).dot(y)
    }).to_host();
    println!("{:?}",bate_hat);
}
```
The example code implements the least squares method using the code that is most similar to the data formula.
$$
 (X^TX)^{-1}X^Ty
$$
V.S.
```Rust
x_t.dot(x).inv().dot(&x_t).dot(y)
```

## Safety and Error Handling

This library uses unsafe code to interface with CUDA functions. It includes error handling that checks the status of each CUDA and cuBLAS call, ensuring that any errors are handled gracefully and reported appropriately.

## Performance

The performance test was conducted using the following code, comparing the dot method provided by ndarray-linalg

```Rust
fn dot_with_ndarry() {
    let a = Array::from_elem((H_SIZE, H_SIZE), 1.0_f32);
    let b = Array::from_elem((H_SIZE, V_SIZE), 1.0_f32);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = a.dot(&b);
    }
    println!("ndarray dot elapsed: {:.2?}", start.elapsed());
}

fn dot_with_cuda() {
    let a = Array::from_elem((H_SIZE, H_SIZE), 1.0_f32);
    let b = Array::from_elem((H_SIZE, V_SIZE), 1.0_f32);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = a.cuda_dot(&b);
    }
    println!("matmul elapsed: {:.2?}", start.elapsed());
}
```
Comparing result:


|Rows|columns|run times|ndarra-linalg|cuda_dot|
|----|----|----|----|----|
|64|64|100|2.27ms|9.89ms|
|128|80|100|11.37ms|10.66ms|
|768|128|100|438.01ms|57.86ms|
|2048|1000|100|22800ms|323.30ms|

## Contribution

Contributions to this library are welcome! Whether it's through reporting issues, proposing new features, improving documentation, or submitting pull requests, all forms of contribution are encouraged.

## License

This library is distributed under the MIT license. 

