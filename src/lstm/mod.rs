use candle_core::Result;
use polars::prelude::{CsvReadOptions, CsvReader, SerReader,LazyFrame,LazyCsvReader,LazyFileListReader};

pub fn main() -> Result<()> {
    let df = LazyCsvReader::new("./dataset/ratings_test.csv")
    .with_truncate_ragged_lines(true)
    .with_ignore_errors(true) // 파싱 오류 무시
    .with_rechunk(true)
    .with_skip_rows_after_header(0)
    .finish()
    .unwrap()
    .collect()
    .unwrap();    
println!("{:?}",df.height());
println!("{:?}",df.head(Some(10)));

    Ok(())
}
