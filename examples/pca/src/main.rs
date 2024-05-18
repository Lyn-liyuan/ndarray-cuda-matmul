use ndarray::Array2;

use ndarray::s;
use ndarray_cuda_matmul::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use plotters::prelude::*;
use polars::io::prelude::*;
use polars::lazy::prelude::*;
use polars_core::prelude::*;

const OUT_FILE_NAME: &str = "dataset/matshow.svg";

fn show_mat(matrix: &Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Cov Matrix", ("sans-serif", 60))
        .margin(5)
        .top_x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0i32..13i32, 13i32..0i32)?;

    chart
        .configure_mesh()
        .x_labels(13)
        .y_labels(13)
        .max_light_lines(4)
        .x_label_offset(35)
        .y_label_offset(25)
        .disable_x_mesh()
        .disable_y_mesh()
        .label_style(("sans-serif", 20))
        .draw()?;

    let range = 0usize..13usize;
    let max = matrix.max().unwrap();
    let min = matrix.min().unwrap();

    chart.draw_series(
        range
            .clone()
            .flat_map(|row| {
                range.clone().map(move |column| {
                    (
                        row as i32,
                        column as i32,
                        matrix.get((row, column)).unwrap(),
                    )
                })
            })
            .map(|(x, y, v)| {
                let value = (*v - min) / (max - min);
                EmptyElement::at((x, y))
                    + Rectangle::new(
                        [(0, 0), (70, 47)],
                        HSLColor(value as f64 + 0.2, 0.8, value as f64 + 0.1).filled(),
                    )
                    + Text::new(
                        format!("{:.3}", *v),
                        (15, 20),
                        ("sans-serif", 15.0).into_font(),
                    )
            }),
    )?;
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

fn scatter(matrix: &Array2<f32>,y:&Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new("dataset/scatter.svg", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let x_max = matrix.max().unwrap();
    let x_min = matrix.min().unwrap();
    let y_max = y.max().unwrap();
    let y_min = y.min().unwrap();
    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(*x_min as f64..*x_max as f64, *y_min as f64..*y_max as f64)?;
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    let x_range = 0usize..506usize;

    scatter_ctx.draw_series(
        x_range
            .map(|x| Circle::new((*matrix.get((x,0)).unwrap() as f64, *y.get((x,13)).unwrap() as f64), 2, GREEN.filled())),
    )?;
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to dataset/scatter.svg");

    Ok(())
}

fn main() {
    let df = CsvReader::from_path("dataset/BostonHousing.csv")
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();

    println!("{:?}", df.get_column_names());
    println!("{}", df.head(Some(5)));
    let standardization = df.lazy().select([col("*") / col("*").max()]);
    let center = standardization
        .select([col("*") - col("*").mean()])
        .collect()
        .unwrap();

    println!("{}", &center);
    let array = center.to_ndarray::<Float32Type>(IndexOrder::C).unwrap();
    let m = array.shape()[0];
    println!("m={}", m);
    let mat = array.slice(s![0..506,0..14]).to_owned();
    let cov_mat = run! {mat => {
        mat.t().dot(mat).mul_scalar(1.0f32/(m as f32-1.0f32))
    }}
    .to_host();

    println!("cov_mat:\n{}", &cov_mat);
    show_mat(&cov_mat).unwrap();

    let (eigs, vecs) = cov_mat.eig().unwrap();
    println!("eigs {}", &eigs);
    println!("vecs {}", &vecs);
    let binding = vecs.map(|v| v.re);
    let evec = binding.slice(s![0..1, ..]);
    let reduced = mat.dot(&evec.t());

    println!("reduced shape {:?}", reduced.shape());
    scatter(&reduced,&array).unwrap();
}
