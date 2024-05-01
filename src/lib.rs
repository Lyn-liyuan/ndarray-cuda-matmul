// External crate declaration for `libc` which provides FFI (Foreign Function Interface) bindings.
extern crate libc;

// Importing necessary components from the `ndarray` crate to work with n-dimensional arrays.
use libc::size_t;
use ndarray::{
    Array, ArrayBase, Dim, Dimension, Ix1, Ix2, IxDynImpl, OwnedRepr, ShapeBuilder, ViewRepr,
};
use std::ffi::c_float;

#[repr(C)]
struct MatParameter {
    data: *mut c_float,
    size: size_t,
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
    fn to_host(out: *mut f32, m: *const MatParameter);
    fn to_device(inp: *const f32, size: usize) -> *mut f32;
    fn cuda_free(m: *mut c_float);
    fn mat_free(m: *mut MatParameter);
    fn matmul_cublas_device(
        a: *const c_float,
        b: *const c_float,
        m: size_t,
        n: size_t,
        k: size_t,
    ) -> *mut MatParameter;

    fn scalar_mul_device(a: *const c_float, scalar: c_float, size: size_t) -> *mut MatParameter;
    fn inv_device(a: *const c_float,n: size_t) -> *mut MatParameter;
}

pub struct CudaMat {
    data: *mut c_float,
    shape: Vec<usize>,
}

impl Drop for CudaMat {
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.data);
        }
    }
}

impl CudaMat {
    pub fn dot(&self, mat: &CudaMat) -> CudaMat {
        let (m, n, k) = if self.shape.len() == 1 || mat.shape.len() == 1 {
            let dim = if mat.shape.len() == 1 {
                mat.shape.get(0).unwrap()
            } else {
                self.shape.get(0).unwrap()
            };
            (1_usize, 1_usize, *dim)
        } else {
            // Otherwise, return the respective dimensions for the matrix multiplication.
            (
                *self.shape.get(0).unwrap(),
                *mat.shape.get(1).unwrap(),
                *self.shape.get(1).unwrap(),
            )
        };

        unsafe {
            let mat_p = Some(matmul_cublas_device(self.data, mat.data, m, n, k));
            let result = CudaMat {
                data: (*(mat_p.unwrap())).data,
                shape: if m == 1 { vec![1_usize] } else { vec![m, n] },
            };

            mat_free(mat_p.unwrap());

            result
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> CudaMat {
        let dim = self.shape.len();
        let size = if dim == 1 {
            *self.shape.get(0).unwrap()
        } else {
            self.shape.get(0).unwrap() * self.shape.get(1).unwrap()
        };
        unsafe {
            let mat_p = scalar_mul_device(self.data, scalar, size);
            let result = CudaMat {
                data: (*mat_p).data,
                shape: if dim == 1 {
                    vec![1_usize]
                } else {
                    vec![*self.shape.get(0).unwrap(), *self.shape.get(1).unwrap()]
                },
            };
            mat_free(mat_p);
            result
        }
    }

    pub fn inv(&self) -> CudaMat {
        let n = *self.shape.get(0).unwrap();
        unsafe {
            let mat_p = inv_device(self.data, n);
            let result = CudaMat {
                data: (*mat_p).data,
                shape: vec![*self.shape.get(0).unwrap(), *self.shape.get(1).unwrap()]
            
            };
            mat_free(mat_p);
            result
        }
    }

    pub fn to_host(&self) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
        let shape = <Vec<usize> as Clone>::clone(&self.shape).into_shape();
        let mut out = ndarray::Array::zeros(shape);
        let mat_p = MatParameter {
            data: self.data,
            size: if self.shape.len() == 1 {
                *self.shape.get(0).unwrap()
            } else {
                *self.shape.get(0).unwrap() * *self.shape.get(1).unwrap()
            },
        };

        unsafe {
            to_host(out.as_mut_ptr(), &mat_p);
        }
        out
    }
}

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
pub trait DeviceDot {
    // The method return a CudaMat hold a pointer of memory in GPU
    fn to_device(&self) -> CudaMat;
}

impl DeviceDot for ArrayBase<OwnedRepr<f32>, Ix1> {
    fn to_device(&self) -> CudaMat {
        let size = self.shape()[0];
        unsafe {
            let out = to_device(self.as_ptr(), size);
            CudaMat {
                data: out,
                shape: vec![size],
            }
        }
    }
}

impl DeviceDot for ArrayBase<OwnedRepr<f32>, Ix2> {
    fn to_device(&self) -> CudaMat {
        let size = self.shape()[0] * self.shape()[1];
        unsafe {
            let out = to_device(self.as_ptr(), size);
            CudaMat {
                data: out,
                shape: vec![self.shape()[0], self.shape()[1]],
            }
        }
    }
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
    const H_SIZE: usize = 128;
    const V_SIZE: usize = 1028;

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

    fn dot_with_device() {
        let a = Array::from_elem((H_SIZE, H_SIZE), 1.0_f32);
        let b = Array::from_elem((H_SIZE, V_SIZE), 1.0_f32);
        let start = Instant::now();
        let a_mat = a.to_device();
        let b_mat = b.to_device();
        for _ in 0..100 {
            let _ = a_mat.dot(&b_mat).to_host();
        }
        println!("device matmul elapsed: {:.2?}", start.elapsed());
    }

    #[test]
    fn ix1_dot_ix1_device() {
        let a = array![1.0_f32, 2.0_f32, 3.0_f32];
        let b = array![1.0_f32, 3.0_f32, 5.0_f32];
        init_cublas();
        let out = a.to_device().dot(&b.to_device()).to_host();
        destory_cublas();
        assert!(out[0] == 22.0f32);
    }

    #[test]
    fn ix1_dot_ix2_device() {
        let a = array![1.0_f32, 2.0_f32, 3.0_f32];
        let b = array![[1.0_f32], [3.0_f32], [5.0_f32]];
        init_cublas();
        let out = a.to_device().dot(&b.to_device()).to_host();
        destory_cublas();
        assert!(out[0] == 22.0f32);
    }
    #[test]
    fn ix2_dot_ix2_device() {
        let a = array![[1.0_f32, 2.0_f32, 3.0_f32], [4.0_f32, 5.0_f32, 6.0_f32]];
        let b = array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32], [5.0_f32, 6.0_f32]];
        let c = array![[1.0f32, 1.0f32], [1.0f32, 1.0f32]];
        init_cublas();
        let out = a
            .to_device()
            .dot(&b.to_device())
            .dot(&c.to_device())
            .mul_scalar(2.0_f32)
            .to_host();
        destory_cublas();
        assert!(
            *out.get([0, 0]).unwrap() == 142.0f32
                && *out.get([0, 1]).unwrap() == 184.0f32
                && *out.get([1, 0]).unwrap() == 142.0f32
                && *out.get([1, 1]).unwrap() == 184.0f32
        );

    }

    #[test]
    fn ix2_inv_device() {
        let a = array![[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
                       [2.0_f32, 3.0_f32, 1.0_f32, 2.0_f32],
                       [1.0_f32, 1.0_f32, 1.0_f32, -1.0_f32],
                       [1.0_f32, 0.0_f32, -2.0_f32, -6.0_f32],
                       ];
        
        init_cublas();
        let out = a.to_device().inv().to_host();
        
        destory_cublas();
        println!("1:{:?}",out);
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
        assert!(
            *out.get((0, 0)).unwrap() == 22.0f32
                && *out.get((0, 1)).unwrap() == 49.0f32
                && *out.get((1, 0)).unwrap() == 28.0f32
                && *out.get((1, 1)).unwrap() == 64.0f32
        );
    }

    #[test]
    fn performance() {
        init_cublas();
        dot_with_cuda();
        dot_with_device();
        destory_cublas();
        dot_with_ndarry();
    }
}
