use ndarray::Array2;
use ndarray_cuda_matmul::{run, run2, MatDot, MatInv, MatT, ToHost};
use ndarray_linalg::Inverse;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

fn gen_data() -> (Array2<f32>, Array2<f32>) {
    // 设置线性模型的参数
    let w = [
        3.0f32, 2.0f32, 1.5f32, 4.2f32, 3.0f32, 6.2f32, 1.7f32, 1.2f32, 0.3f32,
    ]; // 线性系数
    let b = 0.7f32; // 截距项
    let n = 100000000; // 数据点数量

    // 创建随机数生成器
    let mut rng = rand::thread_rng();
    let uniform_dist = Uniform::new(0.0, 10.0); // 输入值的范围
    let normal_dist = Normal::new(0.0, 1.0).unwrap(); // 噪音的分布

    // 初始化矩阵以存储输入数据和目标数据
    let mut data: Array2<f32> = Array2::zeros((n, 10));
    let mut y: Array2<f32> = Array2::zeros((n, 1));

    for i in 0..n {
        let noise = normal_dist.sample(&mut rng);

        //截距项为1，接着是x值，最后是y值
        data[[i, 0]] = 1.0f32; // 添加截距项
        let mut y_hat = 0.0f32;
        for j in 1..10 {
            data[[i, j]] = uniform_dist.sample(&mut rng);
            y_hat += w[j - 1] * data[[i, j]];
        }
        y[[i, 0]] = y_hat + b + noise;
    }
    return (data, y);
}

fn do_ndarray_linalg(x: &Array2<f32>, y: &Array2<f32>) {
    let xt = x.t();

    // 计算 X^T * X
    let xtx = xt.dot(x);

    // 计算 (X^T * X)^(-1)
    let xtx_inv = xtx.inv().unwrap();

    // 计算 (X^T * X)^(-1) * X^T
    let xtx_inv_xt = xtx_inv.dot(&xt);

    // 计算 ((X^T * X)^(-1) * X^T) * y，得到系数 b
    let beta = xtx_inv_xt.dot(y);

    println!("{:?}", beta);
}

fn do_cuda(x: &Array2<f32>, y: &Array2<f32>) {
    let out = run!(x,y => {
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
