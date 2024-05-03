// External crate declaration for `libc` which provides FFI (Foreign Function Interface) bindings.
extern crate libc;

// Importing necessary components from the `ndarray` crate to work with n-dimensional arrays.
use libc::size_t;
use ndarray::{Array, ArrayBase, Dimension, Ix1, Ix2, OwnedRepr, ViewRepr};
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
    fn inv_device(a: *const c_float, n: size_t) -> *mut MatParameter;
}

pub struct Cubals {}

impl Cubals {
    pub fn new()->Self {
        unsafe {
           _init_cublas();
        }
        Cubals{}
    }
}

impl Drop for Cubals {
    fn drop(&mut self) {
       unsafe {
          _destory_cublas();
       }
    }
}


pub struct CudaMat<D: Dimension> {
    data: *mut c_float,
    dim: D,
}

impl<D: Dimension> Drop for CudaMat<D> {
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.data);
        }
    }
}

impl<D: Dimension> CudaMat<D> {
    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }
    pub fn mul_scalar(&self, scalar: f32) -> CudaMat<D> {
        let size = self.dim.size();
        unsafe {
            let mat_p = scalar_mul_device(self.data, scalar, size);
            let result = CudaMat {
                data: (*mat_p).data,
                dim: self.dim.clone(),
            };
            mat_free(mat_p);
            result
        }
    }
}

// A trait that defines a CUDA-based dot product between arrays.
pub trait CudaDot<Rhs> {
    type Output;
    // The method signature for performing the dot product using CUDA.
    fn cuda_dot(&self, rhs: &Rhs) -> Self::Output;
}

pub trait MatDot<Rhs> {
    type Output;
    // The method signature for performing the dot product using CUDA.
    fn dot(&self, mat: &Rhs) -> Self::Output;
}

pub trait MatInv<Rhs> {
    // The method signature for performing the dot product using CUDA.
    fn inv(&self) -> Rhs;
}

impl MatInv<CudaMat<Ix2>> for CudaMat<Ix2> {
    fn inv(&self) -> CudaMat<Ix2> {
        let shape = self.shape();
        if shape[0] != shape[1] {
            panic!("The matrix is ​​not a square matrix and cannot be inverted");
        }
        let n = shape[0];
        unsafe {
            let mat_p = inv_device(self.data, n);
            let result = CudaMat {
                data: (*mat_p).data,
                dim: Ix2(shape[0], shape[1]),
            };
            mat_free(mat_p);
            result
        }
    }
}

impl MatDot<CudaMat<Ix1>> for CudaMat<Ix1> {
    type Output = CudaMat<Ix1>;
    fn dot(&self, mat: &CudaMat<Ix1>) -> Self::Output {
        let shape = self.shape();
        let mat_shape = mat.shape();
        if shape[0] != mat_shape[0] {
            panic!("The rows of matrix A are not equal to the columns of matrix B and cannot be multiplied.");
        }
        let (m, n, k) = (1_usize, 1_usize, shape[0]);
        unsafe {
            let mat_p = Some(matmul_cublas_device(self.data, mat.data, m, n, k));
            let result = CudaMat {
                data: (*(mat_p.unwrap())).data,
                dim: Ix1(1_usize),
            };
            mat_free(mat_p.unwrap());
            result
        }
    }
}

impl MatDot<CudaMat<Ix2>> for CudaMat<Ix1> {
    type Output = CudaMat<Ix1>;
    fn dot(&self, mat: &CudaMat<Ix2>) -> Self::Output {
        let shape = self.shape();
        let mat_shape = mat.shape();
        if shape[0] != mat_shape[0] {
            panic!("The rows of matrix A are not equal to the columns of matrix B and cannot be multiplied.");
        }
        let (m, n, k) = (1_usize, 1_usize, shape[0]);
        unsafe {
            let mat_p = Some(matmul_cublas_device(self.data, mat.data, m, n, k));
            let result = CudaMat {
                data: (*(mat_p.unwrap())).data,
                dim: Ix1(1_usize),
            };
            mat_free(mat_p.unwrap());
            result
        }
    }
}

impl MatDot<CudaMat<Ix1>> for CudaMat<Ix2> {
    type Output = CudaMat<Ix1>;
    fn dot(&self, mat: &CudaMat<Ix1>) -> Self::Output {
        let shape = self.shape();
        let mat_shape = mat.shape();
        if shape[0] != mat_shape[0] {
            panic!("The rows of matrix A are not equal to the columns of matrix B and cannot be multiplied.");
        }
        let (m, n, k) = (1_usize, 1_usize, mat_shape[0]);
        unsafe {
            let mat_p = Some(matmul_cublas_device(self.data, mat.data, m, n, k));
            let result = CudaMat {
                data: (*(mat_p.unwrap())).data,
                dim: Ix1(1_usize),
            };
            mat_free(mat_p.unwrap());
            result
        }
    }
}

impl MatDot<CudaMat<Ix2>> for CudaMat<Ix2> {
    type Output = CudaMat<Ix2>;
    fn dot(&self, mat: &CudaMat<Ix2>) -> Self::Output {
        let shape = self.shape();
        let mat_shape = mat.shape();
        if mat_shape[0] != shape[1] {
            panic!("The rows of matrix A are not equal to the columns of matrix B and cannot be multiplied.");
        }
        let (m, n, k) = (shape[0], mat_shape[1], shape[1]);
        unsafe {
            let mat_p = Some(matmul_cublas_device(self.data, mat.data, m, n, k));
            let result = CudaMat {
                data: (*(mat_p.unwrap())).data,
                dim: Ix2(m, n),
            };
            mat_free(mat_p.unwrap());
            result
        }
    }
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
pub trait ToDevice<D>
where
    D: Dimension,
{
    // The method return a CudaMat hold a pointer of memory in GPU
    fn to_device(&self) -> CudaMat<D>;
}

impl ToDevice<Ix1> for ArrayBase<OwnedRepr<f32>, Ix1> {
    fn to_device(&self) -> CudaMat<Ix1> {
        let size = self.shape()[0];
        unsafe {
            let out = to_device(self.as_ptr(), size);
            CudaMat {
                data: out,
                dim: Ix1(size),
            }
        }
    }
}

impl ToDevice<Ix2> for ArrayBase<OwnedRepr<f32>, Ix2> {
    fn to_device(&self) -> CudaMat<Ix2> {
        let size = self.shape()[0] * self.shape()[1];
        unsafe {
            let out = to_device(self.as_ptr(), size);
            CudaMat {
                data: out,
                dim: Ix2(self.shape()[0], self.shape()[1]),
            }
        }
    }
}

pub trait ToHost<D>
where
    D: Dimension,
{
    // The method return a CudaMat hold a pointer of memory in GPU
    fn to_host(&self) -> ArrayBase<OwnedRepr<f32>, D>;
}

impl ToHost<Ix1> for CudaMat<Ix1> {
    fn to_host(&self) -> ArrayBase<OwnedRepr<f32>, Ix1> {
        let shape = self.shape();
        let mut out = ndarray::Array::from_elem(shape[0], 0.0_f32);
        let mat_p = MatParameter {
            data: self.data,
            size: shape[0],
        };
        unsafe {
            to_host(out.as_mut_ptr(), &mat_p);
        }
        out
    }
}

impl ToHost<Ix2> for CudaMat<Ix2> {
    fn to_host(&self) -> ArrayBase<OwnedRepr<f32>, Ix2> {
        let shape = self.shape();
        let mut out = ndarray::Array::from_elem((shape[0], shape[1]), 0.0_f32);
        let mat_p = MatParameter {
            data: self.data,
            size: self.dim.size(),
        };
        unsafe {
            to_host(out.as_mut_ptr(), &mat_p);
        }
        out
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
        if b_shape[0] != a_shape[0] {
            panic!("The rows of matrix A are not equal to the columns of matrix B and cannot be multiplied.");
        }
        let size = if b.ndim() == 1 {
            b_shape[0]
        } else {
            a_shape[0]
        };

        (1, 1, size)
    } else {
        if a_shape[1] != b_shape[0] {
            panic!("The rows of matrix A are not equal to the columns of matrix B and cannot be multiplied.");
        }
        // Otherwise, return the respective dimensions for the matrix multiplication.
        (a_shape[0], b_shape[1], a_shape[1])
    }
}

#[macro_export]
macro_rules! gpu {
    ($mat1:ident$(.dot($mat2:ident))*)=>{
      {
        Cubals::new();  
        $mat1.to_device()$(.dot(&$mat2.to_device()))*
      }
    };
    ($inv:ident.inv())=>{
      {
        Cubals::new();
        $inv.to_device().inv()
      }
    };
    ($smul:ident.mul_scalar($scalar:ident))=>{
      {
        Cubals::new();
        $smul.to_device().mul_scalar($scalar)
      }
    };
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
        let a = array![
            [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
            [2.0_f32, 3.0_f32, 1.0_f32, 2.0_f32],
            [1.0_f32, 1.0_f32, 1.0_f32, -1.0_f32],
            [1.0_f32, 0.0_f32, -2.0_f32, -6.0_f32],
        ];

        init_cublas();
        let out = a.to_device().inv().to_host();

        destory_cublas();
        println!("1:{:?}", out);
    }
    #[test]
    fn ix2_inv_macro() {
        let a = array![
            [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
            [2.0_f32, 3.0_f32, 1.0_f32, 2.0_f32],
            [1.0_f32, 1.0_f32, 1.0_f32, -1.0_f32],
            [1.0_f32, 0.0_f32, -2.0_f32, -6.0_f32],
        ];
        let b = 1.0_f32;

        let out = gpu!(a.mul_scalar(b));

        let out = out.inv();
        
        let _  = gpu!(a.inv());

        let out = out.to_host();
       
        println!("1:{:?}", out);
    }
    #[test]
    fn ix1_dot_ix1() {
        let a = array![1.0_f32, 2.0_f32, 3.0_f32];
        let b = array![1.0_f32, 3.0_f32, 5.0_f32];
        let out = a.cuda_dot(&b);
        assert!(out[0] == 22.0f32);
    }
    #[test]
    fn ix1_dot_ix1_macro() {
        let a = array![1.0_f32, 2.0_f32, 3.0_f32];
        let b = array![1.0_f32, 3.0_f32, 5.0_f32];
        let c = array![1.0_f32];
        let out: CudaMat<Ix1> = gpu!{
            a.dot(b).dot(c)
        };
        let out = out.to_host();
        assert!(out[0] == 22.0f32);
    }
    #[test]
    #[should_panic]
    fn ix1_dot_ix1_unmatch() {
        let a = array![1.0_f32, 2.0_f32, 3.0_f32];
        let b = array![1.0_f32, 3.0_f32];
        let _ = a.cuda_dot(&b);
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
