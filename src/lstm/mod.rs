use candle_core::Result;
use polars::prelude::{CsvReadOptions, CsvReader, SerReader,LazyFrame,LazyCsvReader,LazyFileListReader};

pub fn main() -> Result<()> {
    let lf = LazyCsvReader::new("./dataset/ratings_train.csv").with_truncate_ragged_lines(true).finish().unwrap().collect().unwrap();

    // let a = CsvReadOptions::default()
    //     .with_has_header(true)
    
    //     .try_into_reader_with_file_path(Some("./dataset/ratings_train.csv".into()))
        
    //     .unwrap()
        
    //     .finish()
    //     .unwrap();
    println!("{:?}",lf);
    Ok(())
}
