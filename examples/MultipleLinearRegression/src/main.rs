use ndarray::Array2;
use ndarray_cuda_matmul::{MatDot, MatInv, MatT, ToHost};
use ndarray_linalg::Inverse;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

fn gen_data() -> (Array2<f32>, Array2<f32>) {
    // Set parameters for the linear model
    let w = [
        3.0f32, 2.0f32, 1.5f32, 4.2f32, 3.0f32, 6.2f32, 1.7f32, 1.2f32, 0.3f32,
    ]; // Linear coefficients
    let b = 0.7f32; // Intercept term
    let n = 100000000; // Number of data points

    // Create random number generator
    let mut rng = rand::thread_rng();
    let uniform_dist = Uniform::new(0.0, 10.0); // Range for input values
    let normal_dist = Normal::new(0.0, 1.0).unwrap(); // Distribution for noise

    // Initialize matrices to store input data and target data
    let mut data: Array2<f32> = Array2::zeros((n, 10));
    let mut y: Array2<f32> = Array2::zeros((n, 1));

    for i in 0..n {
        let noise = normal_dist.sample(&mut rng);

        // The first entry is for the intercept, followed by x values, and finally the y value
        data[[i, 0]] = 1.0f32; // Add the intercept term
        let mut y_hat = 0.0f32;
        for j in 1..10 {
            // Generate random x values
            data[[i, j]] = uniform_dist.sample(&mut rng);
            // Calculate the linear combination of x values and weights
            y_hat += w[j - 1] * data[[i, j]];
        }
        // Assign the calculated y value with added noise
        y[[i, 0]] = y_hat + b + noise;
    }
    return (data, y);
}


fn do_ndarray_linalg(x: &Array2<f32>, y: &Array2<f32>) {
    let xt = x.t();

    // calculate X^T * X
    let xtx = xt.dot(x);

    // calculate (X^T * X)^(-1)
    let xtx_inv = xtx.inv().unwrap();

    // calculate (X^T * X)^(-1) * X^T
    let xtx_inv_xt = xtx_inv.dot(&xt);

    // calculate ((X^T * X)^(-1) * X^T) * y，to get coefficient b
    let beta = xtx_inv_xt.dot(y);

    println!("{:?}", beta);
}

fn do_cuda(x: &Array2<f32>, y: &Array2<f32>) {
    let out = run!(x,y => {
        // (X^TX)^{-1}X^Ty
        let x_t = x.t();
        x_t.dot(x).inv().dot(&x_t).dot(y)
    })
    .to_host();
    println!("{:?}", out);
}
fn main() {
    let (x, y) = gen_data();
    let start = Instant::now();
    do_cuda(&x, &y);
    println!("cuda elapsed: {:.2?}", start.elapsed());
    let start = Instant::now();
    do_ndarray_linalg(&x, &y);
    println!("linalg elapsed: {:.2?}", start.elapsed());
}
