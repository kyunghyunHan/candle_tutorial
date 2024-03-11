use std::ops;

use candle_core::{Device, IndexOp, Tensor};

pub fn main()->anyhow::Result<()>{
    let device = Device::Cpu;

    let x = Tensor::new(&[0., 1., 2., 3., 4., 5., 6.], &device)?;
    println!("{}",x);
    println!("{:?}",x.rank());//차원을 확인
    println!("{:?}",x.shape());//크기를 확인

    //1차원 
 
    println!("t[0]: {:?}", x.i(0).unwrap());
    println!("t[1]: {:?}", x.i(1).unwrap());
    //슬라이싱
    println!("t[1]: {:?}", x.i(2..5).unwrap());
    println!("t[1]: {:?}", x.i(4..6).unwrap());
    println!("t[1]: {:?}", x.i(4..6).unwrap());

    let x = Tensor::new(&[[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
        [10., 11., 12.]
       ], &device)?;
    println!("{}",x);
    println!("{:?}",x.rank());//차원을 확인
    println!("{:?}",x.shape());//크기를 확인
    println!("t[1]: {}", x.i((..,1)).unwrap());
    println!("t[1]: {}", x.i((..,..2)).unwrap());

    //broadcasting
    let m1 = Tensor::new(&[[3., 3.]], &device)?;
    let m2 = Tensor::new(&[[2., 2.]], &device)?;
    let m3=(&m1+&m2).unwrap().i(0).unwrap();
    println!("{:?}",m3);

    
    let m1 = Tensor::new(&[[1., 2.]], &device)?;
    let m2 = Tensor::new(&[[3., 3.]], &device)?;
    
    println!("{:?}",(&m1 + &m2).unwrap());


    let m1 = Tensor::new(&[[1., 2.]], &device)?;
    let m2 = Tensor::new(&[[3.], [4.]], &device)?;


    let m1 = Tensor::new(&[[1., 2.], [3., 4.]], &device)?;
    let m2 = Tensor::new(&[[1.], [2.]], &device)?;
    println!("Shape of Matrix 1:{:?}",m1.shape());
    println!("Shape of Matrix 2:{:?}",m1.shape());
    println!("{}",m1.matmul(&m2).unwrap());
    Ok(())
}