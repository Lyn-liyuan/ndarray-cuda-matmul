use polars::prelude::*;

fn main() {
    let df = CsvReader::from_path("dataset/BostonHousing.csv")
    .unwrap()
    .finish()
    .unwrap();
}
