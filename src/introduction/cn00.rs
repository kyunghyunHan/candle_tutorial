use candle_core::{Device, IndexOp, Tensor};

pub fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;
    // let device = Device::Cpu;

    let x: Tensor = Tensor::new(&[0., 1., 2., 3., 4., 5., 6.], &device)?;
    println!("{}", x);
    println!("{:?}", x.rank()); //차원을 확인
    println!("{:?}", x.shape()); //크기를 확인




    println!("t[0]: {:?}", x.i(0).unwrap());
    println!("t[1]: {:?}", x.i(1).unwrap());
    //슬라이싱
    println!("t[1]: {:?}", x.i(2..5).unwrap());
    println!("t[1]: {:?}", x.i(4..6).unwrap());
    println!("t[1]: {:?}", x.i(4..6).unwrap());

    let x = Tensor::new(
        &[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]],
        &device,
    )?;
    println!("{}", x);
    println!("{:?}", x.rank()); //차원을 확인
    println!("{:?}", x.shape()); //크기를 확인
    // println!("t[1]: {}", x.i((.., 1)).unwrap());
    // println!("t[1]: {}", x.i((.., ..2)).unwrap());

    // //broadcasting
    let m1 = Tensor::new(&[[3., 3.]], &device)?;
    let m2 = Tensor::new(&[[2., 2.]], &device)?;
    // let m3 = (&m1 + &m2).unwrap().i(0).unwrap();
    println!("broadcasting");
    // println!("{:?}", m3);

    let m1 = Tensor::new(&[[1f64, 2f64]], &device)?;
    let m2 = Tensor::new(&[3f64], &device)?;
    println!("{}", (&m1.broadcast_add(&m2).unwrap()));

    // //곱셈
    // let m1 = Tensor::new(&[[1., 2.], [3., 4.]], &device)?;
    // let m2 = Tensor::new(&[[1.], [2.]], &device)?;
    // println!("Shape of Matrix 1:{:?}", m1.shape());
    // println!("Shape of Matrix 2:{:?}", m1.shape());
    // println!("{}", m1.matmul(&m2).unwrap()); //2x1 //행렬곱

    // let m1 = Tensor::new(&[[1., 2.], [3., 4.]], &device)?;
    // let m2 = Tensor::new(&[[1., 2.], [3., 4.]], &device)?;
    // println!("Shape of Matrix 1:{:?}", m1.shape());
    // println!("Shape of Matrix 2:{:?}", m1.shape());
    // // println!("{}",(m1 *m2)?);//2x1  //=>동일한 위차만 곱셈
    // println!("{}", (m1.broadcast_mul(&m2))?); //2x1  //=>동일한 위차만 곱셈

    // let t = Tensor::new(&[1., 2.], &device)?;
    // println!("{}", t.mean(0)?);

    // let t = Tensor::new(&[[1., 2.], [3., 4.]], &device)?;
    // //mean
    // println!("{}", t.mean(0)?);
    // println!("{}", t.mean(1)?);
    // //sum
    // println!("{}", t.sum(0)?);
    // println!("{}", t.sum(1)?);
    // //max argmax
    // println!("{}", t.max(0)?);
    // println!("{}", t.argmax(0)?);
    // println!("{}", t.max(1)?);
    // println!("{}", t.argmax(1)?);

    // let t = Tensor::new(&[[[0., 1., 2.], [3., 4., 5.]], [[6., 7., 8.], [9., 10., 11.]]], &device)?;
    // println!("{:?}",t.shape());

    // println!("{:?}",t.reshape((4,3)));
    // //3차원 텐서의 크기변경
    // println!("{:?}",t.reshape((4,1,3)));
    

    // let t = Tensor::new(&[[0.], [1.], [2.]], &device)?;
    // println!("{:?}",t.shape());
    // println!("{:?}",t.squeeze(1));
    // println!("{:?}",t.squeeze(1)?.shape());
    
    // let t = Tensor::new(&[0., 1., 2.], &device)?;
    
    // println!("{}",t.unsqueeze(0)?);

    Ok(())
}
