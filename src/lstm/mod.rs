use candle_core::Result;
use polars::prelude::*;

pub fn main() -> Result<()> {
    // let df = LazyCsvReader::new("./dataset/ratings_test.csv")
    // .with_truncate_ragged_lines(true)
    // .with_ignore_errors(true) // 파싱 오류 무시
    // .with_rechunk(true)
    // .with_skip_rows_after_header(0)
    // .finish()
    // .unwrap()
    // .collect()
    // .unwrap();
    let df_csv = CsvReadOptions::default()
        .with_infer_schema_length(None)
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_truncate_ragged_lines(true))
        .try_into_reader_with_file_path(Some("dataset/ratings_train.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    println!("{}", df_csv.height());
    println!("{:?}",df_csv.shape());
    println!("{:?}",df_csv.head(Some(3)));

    Ok(())
}
