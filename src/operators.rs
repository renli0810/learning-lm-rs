use std::{
    cmp::{max, min},
    vec,
};

use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let xshape = x.shape();
    let xdata = x.data();
    let ydata = unsafe { y.data_mut() };
    let wdata = w.data();
    let wshape = w.shape();

    let n = xshape[xshape.len() - 1];
    assert!(n == wshape[0]);
    let outersize: usize = xshape.iter().take(xshape.len() - 1).product();
    for i in 0..outersize {
        let offset = i * n;
        let sum_x: f32 = xdata[offset..offset + n].iter().map(|&i| i * i).sum();
        let rms = (sum_x / n as f32 + epsilon).sqrt();
        for (j, (&xval, &wval)) in xdata[offset..offset + n]
            .iter()
            .zip(wdata[0..n].iter())
            .enumerate()
        {
            ydata[offset + j] = wval * xval / rms;
        }
    }
    // todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    for (i, &xval) in _x[0..len].iter().enumerate() {
        _y[i] = _y[i] * xval / (1.0 + (-xval).exp());
    }
    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
}

fn shape_extend(shape: &[usize], target_len: usize) -> Vec<usize> {
    println!("shape:{:?},target_len:{}", shape, target_len);
    let mut res = vec![1; target_len - shape.len()];
    res.extend(shape);
    res
}
fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let alen = a.len();
    let blen = b.len();
    assert!(alen == blen);

    let mut res = Vec::with_capacity(alen);
    for (&adim, &bdim) in a.iter().zip(b.iter()) {
        if adim == bdim {
            res.push(adim);
        } else if adim == 1 {
            res.push(bdim);
        } else if bdim == 1 {
            res.push(adim);
        } else {
            return None;
        }
    }
    Some(res)
}
fn stride_compute(veccompute: &[usize], row: usize, col: usize) -> Vec<usize> {
    let mut strides = Vec::with_capacity(veccompute.len());
    let mut product = 1;
    for &dim in veccompute.iter().rev() {
        strides.push(product * row * col);
        product *= dim;
    }
    strides.reverse();
    strides
}
fn stride_compute_all(veccompute: &[usize]) -> Vec<usize> {
    let mut stride = vec![1; veccompute.len()];
    if veccompute.len() == 0 {
        return stride;
    }
    for i in (0..veccompute.len() - 1).rev() {
        stride[i] = stride[i + 1] * veccompute[i + 1];
    }
    return stride;
}
// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let ashape = a.shape().clone();
    let adims = ashape.len();
    assert!(adims >= 2);
    let m = ashape[adims - 2];
    let p_a = ashape[adims - 1];

    let bshape = b.shape().clone();
    let bdims = bshape.len();
    assert!(bdims >= 2);
    let n = bshape[bdims - 2];
    let p_b = bshape[bdims - 1];

    assert!(p_a == p_b);

    let maxdim = max(adims, bdims);
    let ashape_extend = shape_extend(&ashape[..adims - 2], maxdim - 2);
    let bshape_extend = shape_extend(&bshape[..bdims - 2], maxdim - 2);

    println!(
        "ashape_extend:{:?}bshape_extend:{:?}",
        ashape_extend, bshape_extend
    );
    let cshape_broadcast =
        broadcast_shape(&ashape_extend, &bshape_extend).expect("Broadcast Error!");

    let ashape_stride = stride_compute(&ashape_extend, m, p_a);
    let ashape_stride_broadcast: Vec<usize> = ashape_extend
        .iter()
        .zip(cshape_broadcast.iter())
        .zip(ashape_stride)
        .map(|((&adim, &cdim), stride)| if adim == 1 && cdim != 1 { 0 } else { stride })
        .collect();

    let bshape_stride = stride_compute(&bshape_extend, p_b, n);
    let bshape_stride_broadcast: Vec<usize> = bshape_extend
        .iter()
        .zip(cshape_broadcast.iter())
        .zip(bshape_stride)
        .map(|((&bdim, &cdim), stride)| if bdim == 1 && cdim != 1 { 0 } else { stride })
        .collect();

    let mut cshape = cshape_broadcast.clone();
    cshape.push(m);
    cshape.push(n);
    let cshape_stride = stride_compute_all(&cshape);

    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();

    let mut done_index = vec![0; cshape_broadcast.len()];
    loop {
        let aoffset: usize = done_index
            .iter()
            .zip(ashape_stride_broadcast.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        let boffset: usize = done_index
            .iter()
            .zip(bshape_stride_broadcast.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        let coffset: usize = done_index
            .iter()
            .zip(&cshape_stride[..cshape_broadcast.len()])
            .map(|(&i, &s)| i * s)
            .sum();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..p_a {
                    let a_index = aoffset + i * p_a + k;
                    let b_index = boffset + j * p_a + k;
                    sum += _a[a_index] * _b[b_index] * alpha;

                    //println!("a_index:{} a_data:{} bindex:{} b_data:{}",a_index,_a[a_index],b_index,_b[b_index]);
                }
                let c_index = coffset + i * n + j;
                _c[c_index] = sum + beta * _c[c_index];
                //println!("coffset:{} i:{} j:{} c_index:{} _c[c_index]:{}",coffset,i,j,c_index,_c[c_index]);
            }
        }

        let mut flag = true;
        for i in (0..cshape_broadcast.len()).rev() {
            if flag {
                done_index[i] += 1;
                if done_index[i] >= cshape_broadcast[i] {
                    done_index[i] = 0;
                    flag = true;
                } else {
                    flag = false;
                }
            }
        }
        if flag {
            break;
        }
    }
    c.reshape(&cshape);
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

// #[test]
// fn test_matmul_other() {
//     let mut c = Tensor::<f32>::new(
//         vec![
//             1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
//             20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
//         ],
//         &vec![4, 2, 2, 2],
//     );
//     let a = Tensor::<f32>::new(
//         vec![
//             1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
//             20., 21., 22., 23., 24.,
//         ],
//         &vec![4, 1, 2, 3],
//     );
//     let b = Tensor::<f32>::new(
//         vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
//         &vec![2, 2, 3],
//     );
//     matmul_transb(&mut c, 0.0, &a, &b, 1.);
//     assert!(c.close_to(
//         &Tensor::<f32>::new(
//             vec![
//                 14., 32., 32., 77., 50., 68., 122., 167., 50., 122., 68., 167., 194., 266., 266.,
//                 365., 86., 212., 104., 257., 338., 464., 410., 563., 122., 302., 140., 347., 482.,
//                 662., 554., 761.
//             ],
//             &vec![4, 2, 2, 2]
//         ),
//         1e-3
//     ));
// }
