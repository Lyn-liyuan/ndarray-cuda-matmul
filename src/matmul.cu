#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.h"
#include <mutex>
#include <atomic>

static cublasHandle_t cublas_handle = nullptr;
static std::atomic<int> refCounter(0);
std::mutex mtx;

static const int threadsPerBlock = 256;

extern "C" void _init_cublas()
{
    if (cublas_handle == nullptr)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (cublas_handle == nullptr)
        {
            cublasCheck(cublasCreate(&cublas_handle));
            cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        }
    }
    refCounter.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void _destory_cublas()
{
    if (cublas_handle != nullptr && refCounter.load(std::memory_order_relaxed) > 0)
    {
        refCounter.fetch_sub(1, std::memory_order_relaxed);
        if (refCounter.load(std::memory_order_relaxed) == 0)
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (cublas_handle != nullptr)
            {
                cublasCheck(cublasDestroy(cublas_handle));
                cublas_handle = nullptr;
            }
        }
    }
}

extern "C" struct Mat
{
    float *data;
    size_t size;
};

typedef struct Mat Mat;

extern "C" void to_host(float *out, Mat *m)
{
    cudaCheck(cudaMemcpy(out, m->data, m->size * sizeof(float), cudaMemcpyDeviceToHost));
}

extern "C" float *to_device(float *in, size_t size)
{
    float *out;
    cudaCheck(cudaMalloc(&out, size * sizeof(float)));                            // Allocate GPU memory
    cudaCheck(cudaMemcpy(out, in, size * sizeof(float), cudaMemcpyHostToDevice)); // Corrected size

    return out;
}

extern "C" void cuda_free(float *m)
{
    if (m)
        cudaCheck(cudaFree(m));
}

extern "C" void mat_free(Mat *m)
{
    if (m)
    {
        delete m;
    }
}



extern "C" Mat *t_cublas_device(const float *a, int m, int n)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *out_mat;
    cudaCheck(cudaMalloc(&out_mat, m * n * sizeof(float)));
    cublasCheck(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, a, n, &beta, a, n, out_mat, m));
    cudaCheck(cudaGetLastError());
    Mat *mat = new Mat();
    mat->data = out_mat;
    mat->size = m * n;
    return mat;
}

extern "C" Mat *matmul_cublas_device(const float *a, const float *b,
                                     int m, int n, int k)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *out_mat;
    cudaCheck(cudaMalloc(&out_mat, m * n * sizeof(float)));
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, a, k, b, n, &beta, out_mat, m));
    cudaCheck(cudaGetLastError());
    auto out_mat_t = t_cublas_device(out_mat,n,m);
    cudaCheck(cudaFree(out_mat));
    
    return out_mat_t;
}

extern "C" void matmul_cublas(float *out,
                              const float *a, const float *b,
                              int m, int n, int k)
{

    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *a_mat, *b_mat, *out_mat;
    cudaCheck(cudaMalloc(&a_mat, m * k * sizeof(float)));
    cudaCheck(cudaMalloc(&b_mat, n * k * sizeof(float)));
    cudaCheck(cudaMalloc(&out_mat, m * n * sizeof(float)));
    cudaCheck(cudaMemcpy(a_mat, a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b_mat, b, n * k * sizeof(float), cudaMemcpyHostToDevice));

    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, a_mat, k, b_mat, n, &beta, out_mat, m));
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(out, out_mat, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    if (a_mat)
        cudaCheck(cudaFree(a_mat));
    if (b_mat)
        cudaCheck(cudaFree(b_mat));
    if (out_mat)
        cudaCheck(cudaFree(out_mat));
}

__global__ void scalar_mul_device_kernel(float *out, float *mat, float scalar, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        out[idx] = mat[idx] * scalar;
    }
}

extern "C" Mat *scalar_mul_device(float *mat, float scalar, int size)
{
    float *cuda_out;
    cudaCheck(cudaMalloc(&cuda_out, size * sizeof(float)));

    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalar_mul_device_kernel<<<blocksPerGrid, threadsPerBlock>>>(cuda_out, mat, scalar, size);
    cudaCheck(cudaGetLastError());
    Mat *out_mat = new Mat();
    out_mat->data = cuda_out;
    out_mat->size = size;
    return out_mat;
}

extern "C" Mat *inv_device(float *mat, int n)
{
    float *cuda_out;
    float **cuda_out_array;
    float **cuda_a_array;
    int *p, *info;
    const int batch = 1;
    cudaCheck(cudaMalloc(&cuda_out, n * n * sizeof(float)));
    cudaCheck(cudaMalloc(&cuda_out_array, sizeof(float *)));
    cudaCheck(cudaMalloc(&cuda_a_array, sizeof(float *)));
    cudaCheck(cudaMalloc(&p, n * sizeof(int)));
    cudaCheck(cudaMalloc(&info, sizeof(int)));
    float *a_array[] = {mat};
    float *out_array[] = {cuda_out};
    cudaCheck(cudaMemcpy(cuda_out_array, out_array, sizeof(float *), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(cuda_a_array, a_array, sizeof(float *), cudaMemcpyHostToDevice));
    cublasCheck(cublasSgetrfBatched(cublas_handle, n, cuda_a_array, n, p, info, batch));
    cublasCheck(cublasSgetriBatched(cublas_handle, n, (const float **)cuda_a_array, n, p, cuda_out_array, n, info, batch));
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaFree(p));
    cudaCheck(cudaFree(info));
    cudaCheck(cudaFree(cuda_out_array));
    cudaCheck(cudaFree(cuda_a_array));
    Mat *out_mat = new Mat();
    out_mat->data = cuda_out;
    out_mat->size = n * n;
    return out_mat;
}