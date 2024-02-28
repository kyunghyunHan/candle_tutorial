use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module};

pub fn c4() -> Result<()> {
    //창조
    let device = Device::Cpu;
    //torch.Tensor([[1, 2], [3, 4]])
    let x = Tensor::new(&[[1f32, 2.], [3., 4.]], &device)?;
    println!("{}", x);

    //torch.zeros((2,2))
    let y = Tensor::zeros((2, 2), DType::F32, &device)?;
    println!("{}", y);

    //tensor[:,:4]
    let a = x.i((.., ..1))?;
    println!("{}", a);

    //tensor.view((2,2))
    let b = x.reshape((2, 2))?;
    println!("{}", b);

    //a.matmul(b)
    println!("{}", x.matmul(&b)?);

    //a+b
    println!("{:?}", (&x + &b)?);

    //tensor.to(device="cuda")
    // let c= y.to_device(&Device::new_cuda(0)?)?;

    //tensor.to(dtype=torch.float16)
    let d = y.to_dtype(DType::F16)?;
    println!("{:?}", d);

    //weights = torch.load("model.bin")
    // let test = candle_core::safetensors::load("model.safetensors", &device);
    Ok(())
}
