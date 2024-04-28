// External crate declaration for `libc` which provides FFI (Foreign Function Interface) bindings.
extern crate libc;

// Importing necessary components from the `ndarray` crate to work with n-dimensional arrays.
use ndarray::{Array, ArrayBase, Dimension, Ix1, Ix2, OwnedRepr, ViewRepr};
use std::ffi::c_float;
use libc::size_t;

// A trait that defines a CUDA-based dot product between arrays.
pub trait CudaDot<Rhs> {
    type Output;
    // The method signature for performing the dot product using CUDA.
    fn cuda_dot(&self, rhs: &Rhs) -> Self::Output;
}

// Implementation of CudaDot for 1D owned representation arrays.
impl CudaDot<ArrayBase<OwnedRepr<f32>, Ix1>> for ArrayBase<OwnedRepr<f32>, Ix1> {
    type Output = ArrayBase<OwnedRepr<f32>, Ix1>;

    // Performs dot product on 1D arrays using CUDA and returns the result as a 1-element array.
    fn cuda_dot(&self, rhs: &ArrayBase<OwnedRepr<f32>, Ix1>) -> Self::Output {
        let mut out = Array::from_elem(1, 0.0_f32);
        matmul(&mut out, &self.view(), &rhs.t());
        return out;
    }
}

// Implementation of CudaDot for multiplying a 1D array with a 2D array.
impl CudaDot<ArrayBase<OwnedRepr<f32>, Ix2>> for ArrayBase<OwnedRepr<f32>, Ix1> {
    type Output = ArrayBase<OwnedRepr<f32>, Ix1>;

    // Performs multiplication of a 1D array with a 2D array.
    fn cuda_dot(&self, rhs: &ArrayBase<OwnedRepr<f32>, Ix2>) -> Self::Output {
        let mut out = Array::from_elem(1, 0.0_f32);
        matmul(&mut out, &self.view(), &rhs.view());
        return out;
    }
}

// Implementation of CudaDot for multiplying a 2D array with a 1D array.
impl CudaDot<ArrayBase<OwnedRepr<f32>, Ix1>> for ArrayBase<OwnedRepr<f32>, Ix2> {
    type Output = ArrayBase<OwnedRepr<f32>, Ix1>;

    // Performs multiplication of a 2D array with a 1D array.
    fn cuda_dot(&self, rhs: &ArrayBase<OwnedRepr<f32>, Ix1>) -> Self::Output {
        let mut out = Array::from_elem(1, 0.0_f32);
        matmul(&mut out, &rhs.view(), &self.view());
        return out;
    }
}

// Implementation of CudaDot for 2D arrays.
impl CudaDot<ArrayBase<OwnedRepr<f32>, Ix2>> for ArrayBase<OwnedRepr<f32>, Ix2> {
    type Output = ArrayBase<OwnedRepr<f32>, Ix2>;

    // Performs dot product on two 2D arrays using CUDA and returns a new 2D array.
    fn cuda_dot(&self, rhs: &ArrayBase<OwnedRepr<f32>, Ix2>) -> Self::Output {
        let (m, n, _) = get_shape(&self.view(), &rhs.view());
        let mut out = Array::from_elem((n, m), 0.0_f32);
        matmul(&mut out, &self.view(), &rhs.view());
        return out;
    }
}

// Extern block defining functions implemented in foreign code (e.g. C/C++ using CUDA).
extern "C" {
    fn matmul_cublas(
        out: *mut c_float,
        a: *const c_float,
        b: *const c_float,
        m: size_t,
        n: size_t,
        k: size_t,
    );
    fn _init_cublas();
    fn _destory_cublas();
}

// Wrapper function to initialize CUDA for matrix operations.
pub fn init_cublas() {
    unsafe {
        _init_cublas();
    }
}

// Wrapper function to destroy/free CUDA resources.
pub fn destory_cublas() {
    unsafe { _destory_cublas() }
}

// Function to perform matrix multiplication using the cuBLAS library.
pub fn matmul<D1: Dimension, D2: Dimension, D3: Dimension>(
    out: &mut ArrayBase<OwnedRepr<f32>, D1>,
    a: &ArrayBase<ViewRepr<&f32>, D2>,
    b: &ArrayBase<ViewRepr<&f32>, D3>,
) {
    let out_ptr = out.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let (m, n, k) = get_shape(a, b);
    unsafe {
        _init_cublas();
        matmul_cublas(out_ptr, a_ptr, b_ptr, m, n, k); // Calling the foreign CUDA function.
        _destory_cublas();
    }
}

// Helper function to determine the shape of the resulting matrix after multiplication.
fn get_shape<D2: Dimension, D3: Dimension>(
    a: &ArrayBase<ViewRepr<&f32>, D2>,
    b: &ArrayBase<ViewRepr<&f32>, D3>,
) -> (usize, usize, usize) {
    let a_shape = a.shape();
    let b_shape = b.shape();
    // Handling for the case where either input is a 1D array.
    if a.ndim() == 1 || b.ndim() == 1 {
        let dim = if b.ndim() == 1 {
            b_shape[0]
        } else {
            a_shape[0]
        };
        (1, 1, dim)
    } else {
        // Otherwise, return the respective dimensions for the matrix multiplication.
        (a_shape[0], b_shape[1], a_shape[1])
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;
    use ndarray::Array;
    use std::time::Instant;
    const H_SIZE: usize = 2048;
    const V_SIZE: usize = 1000;

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
    #[test]
    fn ix1_dot_ix1() {
        let a = array![1.0_f32, 2.0_f32, 3.0_f32];
        let b = array![1.0_f32, 3.0_f32, 5.0_f32];
        let out = a.cuda_dot(&b);
        assert!(out[0] == 22.0f32);
    }
    #[test]
    fn ix1_dot_ix2() {
        let a = array![1.0_f32, 2.0_f32, 3.0_f32];
        let b = array![[1.0_f32], [3.0_f32], [5.0_f32]];
        let out = a.cuda_dot(&b);
        assert!(out[0] == 22.0f32);
    }
    #[test]
    fn ix2_dot_ix1() {
        let b = array![1.0_f32, 2.0_f32, 3.0_f32];
        let a = array![[1.0_f32], [3.0_f32], [5.0_f32]];
        let out = a.cuda_dot(&b);
        assert!(out[0] == 22.0f32);
    }
    #[test]
    fn ix2_dot_ix2() {
        let a = array![[1.0_f32, 2.0_f32, 3.0_f32], [4.0_f32, 5.0_f32, 6.0_f32]];
        let b = array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32], [5.0_f32, 6.0_f32]];
        let out = a.cuda_dot(&b);
        assert!(*out.get((0,0)).unwrap() == 22.0f32 && 
                *out.get((0,1)).unwrap() == 49.0f32  &&
                *out.get((1,0)).unwrap() == 28.0f32  &&
                *out.get((1,1)).unwrap() == 64.0f32 );
    }

    #[test]
    fn performance() {
        init_cublas();
        dot_with_cuda();
        destory_cublas();
        dot_with_ndarry();
    }
}
