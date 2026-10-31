//! Tensor operations of the CNN models: the only file of the crate that calls candle.
//!
//! The crate depends on candle at commit 9515d410 of its main branch (ADR 0007).
//! At that commit candle's CPU backend computes 17 of 60 checked backward cases
//! wrongly, through 8 defects (D1 to D8). This file wraps candle's tensor in its
//! own `Tensor` type, so code outside it can only use the functions below. They
//! are either native candle operations whose backward pass we checked, or a
//! replacement (W1 to W8) for one of the defects. Each replacement names its
//! defect and the place in candle's source, as a path inside the candle
//! repository at that commit.
//!
//! Training runs in f32. The gradient check in the tests runs the same
//! functions in f64.
#![forbid(unsafe_code)]

use candle_core::{self as candle, DType, Device};
use candle_nn::{Module, ModuleT};
use rand_chacha::ChaCha8Rng;

use crate::util::{BrutError, Result, normal_f32};

/// A value far below any activation; a max window never picks it.
const PAD_LOW: f64 = -1e30;

/// Weight of the batch statistics in the running statistics of batch norm.
pub const BATCH_NORM_MOMENTUM: f64 = 0.1;

impl From<candle::Error> for BrutError {
    /// Keeps candle's message and drops its backtrace.
    ///
    /// candle captures a backtrace when RUST_BACKTRACE is set. An error text
    /// can reach a log, and a backtrace would make it differ between machines.
    fn from(e: candle::Error) -> Self {
        let mut e = e;
        while let candle::Error::WithBacktrace { inner, .. } = e {
            e = *inner;
        }

        BrutError::Invalid(format!("tensor: {e}"))
    }
}

/// A tensor on the CPU: f32 in training, f64 in the gradient check.
///
/// It wraps candle's tensor. Its field is private, so no code outside this
/// file can call a candle operation on it.
#[derive(Clone, Debug)]
pub struct Tensor(candle::Tensor);

impl Tensor {
    /// A tensor of the given shape, filled with `values` in row-major order.
    pub fn from_f32(values: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        let count: usize = shape.iter().product();
        if values.len() != count {
            return Err(BrutError::Invalid(format!(
                "tensor: {} values do not fill the shape {shape:?}",
                values.len()
            )));
        }

        Ok(Tensor(candle::Tensor::from_vec(
            values,
            shape,
            &Device::Cpu,
        )?))
    }

    /// `std` times standard normal draws from `rng`, in row-major order.
    ///
    /// Replacement W8 for defect D8. candle cannot seed its CPU generator:
    /// `set_seed` fails with "cannot seed the CPU rng with set_seed"
    /// (candle-core/src/cpu_backend/mod.rs:3185). So every random value of a
    /// CNN comes from our own ChaCha8 stream, and none from candle.
    pub fn normal(rng: &mut ChaCha8Rng, shape: &[usize], std: f32) -> Result<Tensor> {
        let count: usize = shape.iter().product();
        let values = (0..count).map(|_| std * normal_f32(rng)).collect();

        Tensor::from_f32(values, shape)
    }

    /// The length of each axis.
    pub fn dims(&self) -> &[usize] {
        self.0.dims()
    }

    /// All values as f32, in row-major order.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        Ok(self
            .0
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?)
    }
}

/// A trainable parameter. The backward pass collects its gradient.
#[derive(Clone, Debug)]
pub struct Param(candle::Var);

impl Param {
    /// A parameter that starts with a copy of `init`.
    pub fn new(init: &Tensor) -> Result<Param> {
        // A detached tensor is never a variable, so from_tensor copies it.
        Ok(Param(candle::Var::from_tensor(&init.0.detach())?))
    }

    /// The current value. Operations on it record the graph for `backward`.
    pub fn tensor(&self) -> Tensor {
        Tensor(self.0.as_tensor().clone())
    }

    /// Overwrites the value with `value`, which must have the same shape.
    pub fn set(&self, value: &Tensor) -> Result<()> {
        if value.dims() != self.0.dims() {
            return Err(BrutError::Invalid(format!(
                "param: a value of shape {:?} does not fit the shape {:?}",
                value.dims(),
                self.0.dims()
            )));
        }

        self.0.set(&value.0)?;
        Ok(())
    }
}

/// The gradients of one backward pass.
#[derive(Debug)]
pub struct Grads(candle::backprop::GradStore);

impl Grads {
    /// The gradient of `param`, or `None` if the loss does not depend on it.
    pub fn get(&self, param: &Param) -> Option<Tensor> {
        self.0.get(param.0.as_tensor()).cloned().map(Tensor)
    }
}

/// Runs the backward pass from a scalar `loss`.
pub fn backward(loss: &Tensor) -> Result<Grads> {
    if !loss.dims().is_empty() {
        return Err(BrutError::Invalid(format!(
            "backward: the loss must be a scalar, not of shape {:?}",
            loss.dims()
        )));
    }

    Ok(Grads(loss.0.backward()?))
}

/// Number of window positions along an axis of `len` cells.
///
/// `padding` zero cells sit on each side. A window covers
/// `dilation * (kernel - 1) + 1` cells and moves by `stride`. Every
/// convolution and pooling below has this output size.
pub fn out_len(
    len: usize,
    padding: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> Result<usize> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return Err(BrutError::Invalid(format!(
            "window: kernel {kernel}, stride {stride} and dilation {dilation} must be at least 1"
        )));
    }

    let span = dilation * (kernel - 1) + 1;
    if len + 2 * padding < span {
        return Err(BrutError::Invalid(format!(
            "window: a window of span {span} does not fit into {len} cells with padding {padding}"
        )));
    }

    Ok((len + 2 * padding - span) / stride + 1)
}

/// Checks that `groups` divides the input and output channels of a convolution.
fn check_groups(
    op: &str,
    c_in: usize,
    c_out: usize,
    c_in_group: usize,
    groups: usize,
) -> Result<()> {
    if groups == 0 || !c_in.is_multiple_of(groups) || !c_out.is_multiple_of(groups) {
        return Err(BrutError::Invalid(format!(
            "{op}: {groups} groups do not divide {c_in} input and {c_out} output channels"
        )));
    }

    if c_in / groups != c_in_group {
        return Err(BrutError::Invalid(format!(
            "{op}: the kernel reads {c_in_group} channels per group, the input has {}",
            c_in / groups
        )));
    }

    Ok(())
}

/// Checks the padding of a pooling window; PyTorch allows at most half the kernel.
///
/// Every window then holds at least one cell of the input.
fn check_pool_padding(op: &str, kernel: usize, padding: usize) -> Result<()> {
    if 2 * padding > kernel {
        return Err(BrutError::Invalid(format!(
            "{op}: padding {padding} is more than half the kernel {kernel}"
        )));
    }

    Ok(())
}

/// Zero padding on both sides of the two spatial axes of an NCHW tensor.
fn pad_spatial(x: &candle::Tensor, padding: usize) -> Result<candle::Tensor> {
    if padding == 0 {
        return Ok(x.clone());
    }

    Ok(x.pad_with_zeros(2, padding, padding)?
        .pad_with_zeros(3, padding, padding)?)
}

/// Takes the positions `start, start + step, ...` (`len` of them) along `dim`.
fn take_strided(
    x: &candle::Tensor,
    dim: usize,
    start: usize,
    step: usize,
    len: usize,
) -> Result<candle::Tensor> {
    if step == 1 {
        return Ok(x.narrow(dim, start, len)?);
    }

    let positions = (0..len)
        .map(|o| u32::try_from(start + o * step))
        .collect::<std::result::Result<Vec<u32>, _>>()
        .map_err(|_| BrutError::Invalid("window: a position does not fit into u32".into()))?;
    let index = candle::Tensor::from_vec(positions, len, x.device())?;

    Ok(x.index_select(&index, dim)?)
}

/// 2-D convolution of `x` (n, c_in, h, w) with `kernel` (c_out, c_in / groups, k_h, k_w).
///
/// `padding` zero cells sit on all four sides. Grouped and depthwise
/// convolutions run through candle's own `groups`; their backward pass is
/// correct.
///
/// Replacement W2 for defect D2. candle's backward pass of a 2-D convolution
/// takes the output padding of its transposed convolution from the height
/// alone and uses it for the width too (candle-core/src/backprop.rs:289). With
/// a stride above 1, when height and width leave different remainders, the
/// input gradient gets the wrong width and the backward pass fails with a
/// shape mismatch. In that case this function pads explicitly and crops the
/// rows and columns that no window reads, so both remainders are zero.
pub fn conv2d(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let (_, c_in, h, w) = x.0.dims4()?;
    let (c_out, c_in_group, k_h, k_w) = kernel.0.dims4()?;
    check_groups("conv2d", c_in, c_out, c_in_group, groups)?;
    out_len(h, padding, k_h, stride, dilation)?;
    out_len(w, padding, k_w, stride, dilation)?;

    // Cells behind the last window, along each axis.
    let rest_h = (h + 2 * padding - dilation * (k_h - 1) - 1) % stride;
    let rest_w = (w + 2 * padding - dilation * (k_w - 1) - 1) % stride;

    if rest_h == rest_w {
        let out = x.0.conv2d(&kernel.0, padding, stride, dilation, groups)?;
        return Ok(Tensor(out));
    }

    let cropped = pad_spatial(&x.0, padding)?
        .narrow(2, 0, h + 2 * padding - rest_h)?
        .narrow(3, 0, w + 2 * padding - rest_w)?;
    let out = cropped.conv2d(&kernel.0, 0, stride, dilation, groups)?;

    Ok(Tensor(out))
}

/// 1-D convolution of `x` (n, c_in, len) with `kernel` (c_out, c_in / groups, k).
///
/// Replacement W1 for defect D1. candle's CPU `conv1d` copies a strided kernel
/// into a contiguous buffer, but then multiplies with the original strided
/// buffer (candle-core/src/cpu_backend/mod.rs:2861-2869). Its backward pass
/// hands the transposed output gradient to `conv1d` as such a strided kernel
/// (candle-core/src/backprop.rs:264), so the kernel gradient is wrong whenever
/// the batch and the output channels per group both exceed 1. This function
/// runs `conv2d` on a height of 1 instead, with the padding done explicitly.
pub fn conv1d(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    x.0.dims3()?;
    kernel.0.dims3()?;

    let padded = if padding == 0 {
        x.0.clone()
    } else {
        x.0.pad_with_zeros(2, padding, padding)?
    };

    let out = conv2d(
        &Tensor(padded.unsqueeze(2)?),
        &Tensor(kernel.0.unsqueeze(2)?),
        0,
        stride,
        dilation,
        groups,
    )?;

    Ok(Tensor(out.0.squeeze(2)?))
}

/// Causal 1-D convolution: zero padding on the left only, so output t reads inputs up to t.
///
/// The stride is 1, so the output is as long as the input. Built on `conv1d`
/// (W1), so it avoids defect D1 as well.
pub fn causal_conv1d(
    x: &Tensor,
    kernel: &Tensor,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let (_, _, k) = kernel.0.dims3()?;
    if k == 0 {
        return Err(BrutError::Invalid(
            "causal_conv1d: the kernel needs at least one tap".into(),
        ));
    }

    let padded = x.0.pad_with_zeros(2, (k - 1) * dilation, 0)?;

    conv1d(&Tensor(padded), kernel, 0, 1, dilation, groups)
}

/// 2-D transposed convolution of `x` (n, c_in, h, w) with `kernel` (c_in, c_out, k_h, k_w).
///
/// The output has `(h - 1) * stride - 2 * padding + dilation * (k_h - 1) +
/// output_padding + 1` rows, and the same rule holds for the columns. This is
/// candle's own operation; its backward pass is correct, also on non-square
/// inputs.
pub fn conv_transpose2d(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    let (_, c_in, h, w) = x.0.dims4()?;
    let (c_in_kernel, _, k_h, k_w) = kernel.0.dims4()?;
    let empty = h == 0 || w == 0 || k_h == 0 || k_w == 0;
    if c_in != c_in_kernel || empty || stride == 0 || dilation == 0 {
        return Err(BrutError::Invalid(format!(
            "conv_transpose2d: input {:?} and kernel {:?} with stride {stride} and dilation \
             {dilation} do not fit",
            x.dims(),
            kernel.dims()
        )));
    }

    if output_padding >= stride.max(dilation) {
        return Err(BrutError::Invalid(format!(
            "conv_transpose2d: output padding {output_padding} must be below the stride \
             {stride} or the dilation {dilation}"
        )));
    }

    for (len, k) in [(h, k_h), (w, k_w)] {
        // The output length before the padding is cut from both ends.
        let full = (len - 1) * stride + dilation * (k - 1) + output_padding + 1;
        if full <= 2 * padding {
            return Err(BrutError::Invalid(format!(
                "conv_transpose2d: padding {padding} leaves no output"
            )));
        }
    }

    let out =
        x.0.conv_transpose2d(&kernel.0, padding, output_padding, stride, dilation)?;

    Ok(Tensor(out))
}

/// Max pooling of `x` (n, c, h, w) with a square window.
///
/// Replacement W3 for defect D3. candle's backward pass of `max_pool2d` exists
/// only when the kernel equals the stride (candle-core/src/backprop.rs:363).
/// Where it exists, it multiplies the gradient by the share of maxima in the
/// window instead of dividing it by their count (backprop.rs:373-374), so the
/// single maximum of a 2x2 window gets a quarter of its gradient. This
/// function takes the running maximum over the `kernel * kernel` shifted views
/// of the padded input. On a tie, candle's `maximum` gives each side half the
/// gradient, so the shares within a window still add up to 1. Padding never
/// wins a window.
pub fn max_pool2d(x: &Tensor, kernel: usize, stride: usize, padding: usize) -> Result<Tensor> {
    let (_, _, h, w) = x.0.dims4()?;
    let h_out = out_len(h, padding, kernel, stride, 1)?;
    let w_out = out_len(w, padding, kernel, stride, 1)?;
    check_pool_padding("max_pool2d", kernel, padding)?;

    let padded = if padding == 0 {
        x.0.clone()
    } else {
        // Zero inside and PAD_LOW in the border; adding it keeps the inside exact.
        let inside = candle::Tensor::ones((1, 1, h, w), x.0.dtype(), x.0.device())?;
        let border = (pad_spatial(&inside, padding)?.affine(-1.0, 1.0)? * PAD_LOW)?;
        pad_spatial(&x.0, padding)?.broadcast_add(&border)?
    };

    let mut out: Option<candle::Tensor> = None;
    for i in 0..kernel {
        let rows = take_strided(&padded, 2, i, stride, h_out)?;
        for j in 0..kernel {
            let tap = take_strided(&rows, 3, j, stride, w_out)?;
            out = Some(match out {
                None => tap,
                Some(running) => running.maximum(&tap)?,
            });
        }
    }

    out.map(Tensor)
        .ok_or_else(|| BrutError::Invalid("max_pool2d: the kernel is empty".into()))
}

/// Average pooling of `x` (n, c, h, w) with a square window.
///
/// Padding counts as zeros, as in PyTorch's default.
///
/// Replacement W4 for defect D4. candle's backward pass of `avg_pool2d` exists
/// only when the kernel equals the stride (candle-core/src/backprop.rs:348).
/// Where it exists, it upsamples the gradient to the full input
/// (backprop.rs:352), which puts it on the wrong cells when the size is not a
/// multiple of the kernel. When the kernel equals the stride and there is no
/// padding, this function crops the input to a multiple of the kernel and then
/// calls candle. Otherwise it sums the `kernel * kernel` shifted views.
pub fn avg_pool2d(x: &Tensor, kernel: usize, stride: usize, padding: usize) -> Result<Tensor> {
    let (_, _, h, w) = x.0.dims4()?;
    let h_out = out_len(h, padding, kernel, stride, 1)?;
    let w_out = out_len(w, padding, kernel, stride, 1)?;
    check_pool_padding("avg_pool2d", kernel, padding)?;

    if kernel == stride && padding == 0 {
        return Ok(Tensor(avg_pool2d_even(&x.0, kernel, h_out, w_out)?));
    }

    Ok(Tensor(avg_pool2d_views(
        &x.0, kernel, stride, padding, h_out, w_out,
    )?))
}

/// Average pooling with the kernel equal to the stride: crop, then candle's own operation.
///
/// After the crop, height and width are multiples of the kernel, so candle's
/// backward pass puts the gradient on the right cells.
fn avg_pool2d_even(
    x: &candle::Tensor,
    kernel: usize,
    h_out: usize,
    w_out: usize,
) -> Result<candle::Tensor> {
    let cropped = x
        .narrow(2, 0, h_out * kernel)?
        .narrow(3, 0, w_out * kernel)?;

    Ok(cropped.avg_pool2d(kernel)?)
}

/// Average pooling from the shifted views of the zero-padded input.
fn avg_pool2d_views(
    x: &candle::Tensor,
    kernel: usize,
    stride: usize,
    padding: usize,
    h_out: usize,
    w_out: usize,
) -> Result<candle::Tensor> {
    let padded = pad_spatial(x, padding)?;

    let mut sum: Option<candle::Tensor> = None;
    for i in 0..kernel {
        let rows = take_strided(&padded, 2, i, stride, h_out)?;
        for j in 0..kernel {
            let tap = take_strided(&rows, 3, j, stride, w_out)?;
            sum = Some(match sum {
                None => tap,
                Some(running) => (running + tap)?,
            });
        }
    }

    match sum {
        Some(sum) => Ok((sum * (1.0 / (kernel * kernel) as f64))?),
        None => Err(BrutError::Invalid("avg_pool2d: the kernel is empty".into())),
    }
}

/// Max pooling of `x` (n, c, len) along the sequence, with a window of `kernel` cells.
///
/// candle has no 1-D pooling. Like `max_pool2d` (W3), this function takes the
/// running maximum over the `kernel` shifted views of the input. Cells behind
/// the last window are dropped.
pub fn max_pool1d(x: &Tensor, kernel: usize, stride: usize) -> Result<Tensor> {
    let (_, _, len) = x.0.dims3()?;
    let out = out_len(len, 0, kernel, stride, 1)?;

    let mut running: Option<candle::Tensor> = None;
    for j in 0..kernel {
        let tap = take_strided(&x.0, 2, j, stride, out)?;
        running = Some(match running {
            None => tap,
            Some(max) => max.maximum(&tap)?,
        });
    }

    running
        .map(Tensor)
        .ok_or_else(|| BrutError::Invalid("max_pool1d: the kernel is empty".into()))
}

/// Average pooling of `x` (n, c, len) along the sequence, with a window of `kernel` cells.
///
/// candle has no 1-D pooling. Like the general path of `avg_pool2d` (W4), this
/// function sums the `kernel` shifted views of the input. Cells behind the last
/// window are dropped.
pub fn avg_pool1d(x: &Tensor, kernel: usize, stride: usize) -> Result<Tensor> {
    let (_, _, len) = x.0.dims3()?;
    let out = out_len(len, 0, kernel, stride, 1)?;

    let mut sum: Option<candle::Tensor> = None;
    for j in 0..kernel {
        let tap = take_strided(&x.0, 2, j, stride, out)?;
        sum = Some(match sum {
            None => tap,
            Some(running) => (running + tap)?,
        });
    }

    match sum {
        Some(sum) => Ok(Tensor((sum * (1.0 / kernel as f64))?)),
        None => Err(BrutError::Invalid("avg_pool1d: the kernel is empty".into())),
    }
}

/// Nearest upsampling of `x` (n, c, h, w) by an integer factor on both axes.
///
/// Replacement W5 for defect D5. candle's backward pass of
/// `upsample_nearest2d` assigns its gradient instead of adding it
/// (`*sum_grad = conv_sum`, candle-core/src/backprop.rs:409; the 1-D version at
/// backprop.rs:388). So a tensor that also feeds another operation can lose
/// the gradient of that other operation. This function broadcasts every pixel
/// into a `factor x factor` block; the backward pass of a broadcast adds.
pub fn upsample_nearest2d(x: &Tensor, factor: usize) -> Result<Tensor> {
    if factor == 0 {
        return Err(BrutError::Invalid(
            "upsample_nearest2d: the factor must be at least 1".into(),
        ));
    }

    let (n, c, h, w) = x.0.dims4()?;
    let out =
        x.0.reshape((n, c, h, 1, w, 1))?
            .broadcast_as((n, c, h, factor, w, factor))?
            .reshape((n, c, h * factor, w * factor))?;

    Ok(Tensor(out))
}

/// The shape (1, c, 1, ...) of a per-channel parameter for an input of `rank` axes.
fn channel_shape(rank: usize, c: usize) -> Vec<usize> {
    let mut shape = vec![1; rank];
    shape[1] = c;
    shape
}

/// Checks that each named tensor holds one value per channel.
fn check_per_channel(op: &str, c: usize, tensors: &[(&str, &Tensor)]) -> Result<()> {
    for (name, tensor) in tensors {
        if tensor.dims() != [c] {
            return Err(BrutError::Invalid(format!(
                "{op}: {name} {:?} must hold one value for each of {c} channels",
                tensor.dims()
            )));
        }
    }

    Ok(())
}

/// Layer norm over the channels (axis 1) of `x` (n, c, ...), separately at each position.
///
/// This is the layer norm of ConvNeXt. The result is scaled by `gamma` and
/// shifted by `beta`, one value per channel.
///
/// Replacement W6 for defect D6. candle_nn's `LayerNorm` runs a fused kernel
/// when its input is contiguous (candle-nn/src/layer_norm.rs:118-120). That
/// kernel is applied with `apply_op3_no_bwd` (candle-nn/src/ops.rs:944), so the
/// gradient stops there, and no error says so. It also has no f64 version
/// (candle-nn/src/ops.rs:772). This function uses plain tensor operations.
pub fn layer_norm_channels(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f64) -> Result<Tensor> {
    let rank = x.dims().len();
    if rank < 2 {
        return Err(BrutError::Invalid(format!(
            "layer_norm_channels: the input {:?} has no channel axis",
            x.dims()
        )));
    }

    let c = x.dims()[1];
    check_per_channel(
        "layer_norm_channels",
        c,
        &[("gamma", gamma), ("beta", beta)],
    )?;

    let mean = x.0.mean_keepdim(1)?;
    let centred = x.0.broadcast_sub(&mean)?;
    let var = centred.sqr()?.mean_keepdim(1)?;
    let normed = centred.broadcast_div(&(var + eps)?.sqrt()?)?;

    let shape = channel_shape(rank, c);
    let out = normed
        .broadcast_mul(&gamma.0.reshape(shape.as_slice())?)?
        .broadcast_add(&beta.0.reshape(shape.as_slice())?)?;

    Ok(Tensor(out))
}

/// Batch norm over the channels (axis 1) of `x` (n, c, ...).
///
/// In training mode it normalizes with the mean and variance of the batch and
/// moves the running statistics towards them by `BATCH_NORM_MOMENTUM`. In
/// evaluation mode it normalizes with the running statistics. This is
/// candle_nn's `BatchNorm`; its gradient in training mode is correct.
#[derive(Clone, Debug)]
pub struct BatchNorm(candle_nn::BatchNorm);

impl BatchNorm {
    /// A batch norm with scale `gamma`, shift `beta` and the given running statistics.
    ///
    /// All four tensors hold one value per channel. A new network starts with
    /// a running mean of 0 and a running variance of 1.
    pub fn new(
        gamma: &Tensor,
        beta: &Tensor,
        running_mean: &Tensor,
        running_var: &Tensor,
        eps: f64,
    ) -> Result<BatchNorm> {
        let c = running_mean.dims().first().copied().unwrap_or(0);
        check_per_channel(
            "batch_norm",
            c,
            &[
                ("gamma", gamma),
                ("beta", beta),
                ("running_mean", running_mean),
                ("running_var", running_var),
            ],
        )?;

        let norm = candle_nn::BatchNorm::new_with_momentum(
            c,
            running_mean.0.detach(),
            running_var.0.detach(),
            gamma.0.clone(),
            beta.0.clone(),
            eps,
            BATCH_NORM_MOMENTUM,
        )?;

        Ok(BatchNorm(norm))
    }

    /// Training mode: batch statistics; the running statistics move towards them.
    pub fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
        // Values per channel: every axis but axis 1.
        let per_channel: usize = x
            .dims()
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != 1)
            .map(|(_, len)| len)
            .product();
        if x.dims().len() < 2 || per_channel < 2 {
            return Err(BrutError::Invalid(format!(
                "batch_norm: the input {:?} needs at least 2 values per channel",
                x.dims()
            )));
        }

        Ok(Tensor(self.0.forward_t(&x.0, true)?))
    }

    /// Evaluation mode: the running statistics.
    pub fn forward_eval(&self, x: &Tensor) -> Result<Tensor> {
        Ok(Tensor(self.0.forward_t(&x.0, false)?))
    }

    /// The running mean, one value per channel.
    pub fn running_mean(&self) -> Tensor {
        Tensor(self.0.running_mean().clone())
    }

    /// The running variance, one value per channel.
    pub fn running_var(&self) -> Tensor {
        Tensor(self.0.running_var().clone())
    }
}

/// Group norm of `x` (n, c, ...): the channels form `groups` groups, each normalized as a whole.
///
/// The result is scaled by `gamma` and shifted by `beta`, one value per
/// channel. This is candle_nn's `GroupNorm`; its gradient is correct.
pub fn group_norm(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    groups: usize,
    eps: f64,
) -> Result<Tensor> {
    if x.dims().len() < 3 || groups == 0 {
        return Err(BrutError::Invalid(format!(
            "group_norm: the input {:?} needs a spatial axis, and groups must be at least 1",
            x.dims()
        )));
    }

    let c = x.dims()[1];
    check_per_channel("group_norm", c, &[("gamma", gamma), ("beta", beta)])?;

    let norm = candle_nn::GroupNorm::new(gamma.0.clone(), beta.0.clone(), c, groups, eps)?;

    Ok(Tensor(norm.forward(&x.0)?))
}

/// Concatenation of `parts` along the axis `dim`; the other axes must agree.
pub fn cat(parts: &[&Tensor], dim: usize) -> Result<Tensor> {
    let inner: Vec<&candle::Tensor> = parts.iter().map(|part| &part.0).collect();

    Ok(Tensor(candle::Tensor::cat(&inner, dim)?))
}

/// Elementwise product; an axis of length 1 is repeated, as in (n, c, h, w) times (n, c, 1, 1).
pub fn broadcast_mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(Tensor(a.0.broadcast_mul(&b.0)?))
}

/// Elementwise sum of two tensors of the same shape, as in a residual connection.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(Tensor((&a.0 + &b.0)?))
}

/// Elementwise sum; an axis of length 1 is repeated, as in a bias (1, c, 1, 1) added to (n, c, h, w).
pub fn broadcast_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(Tensor(a.0.broadcast_add(&b.0)?))
}

/// The channel shuffle of ShuffleNet on `x` (n, c, h, w): channel `g * (c / groups) + i` moves to `i * groups + g`.
pub fn channel_shuffle(x: &Tensor, groups: usize) -> Result<Tensor> {
    let (n, c, h, w) = x.0.dims4()?;
    if groups == 0 || !c.is_multiple_of(groups) {
        return Err(BrutError::Invalid(format!(
            "channel_shuffle: {groups} groups do not divide {c} channels"
        )));
    }

    let out =
        x.0.reshape((n, groups, c / groups, h, w))?
            .transpose(1, 2)?
            .reshape((n, c, h, w))?;

    Ok(Tensor(out))
}

/// The values of `x` (n, ...) as rows (n, rest), in row-major order.
pub fn flatten(x: &Tensor) -> Result<Tensor> {
    if x.dims().len() < 2 {
        return Err(BrutError::Invalid(format!(
            "flatten: the input {:?} has no axis after the batch",
            x.dims()
        )));
    }

    Ok(Tensor(x.0.flatten_from(1)?))
}

/// The values of `x` in row-major order, under a new shape with the same number of values.
pub fn reshape(x: &Tensor, shape: &[usize]) -> Result<Tensor> {
    let count: usize = shape.iter().product();
    if count != x.0.elem_count() {
        return Err(BrutError::Invalid(format!(
            "reshape: {} values do not fill the shape {shape:?}",
            x.0.elem_count()
        )));
    }

    Ok(Tensor(x.0.reshape(shape)?))
}

/// Dense layer: `x` (n, in) times the transposed `weight` (out, in), plus `bias` (out).
///
/// This is candle_nn's `Linear` on a matrix input.
pub fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let fits = match (x.dims(), weight.dims(), bias.dims()) {
        ([_, inputs], [outputs, weight_inputs], [bias_outputs]) => {
            inputs == weight_inputs && outputs == bias_outputs
        }
        _ => false,
    };
    if !fits {
        return Err(BrutError::Invalid(format!(
            "linear: input {:?}, weight {:?} and bias {:?} do not fit",
            x.dims(),
            weight.dims(),
            bias.dims()
        )));
    }

    let layer = candle_nn::Linear::new(weight.0.clone(), Some(bias.0.clone()));

    Ok(Tensor(layer.forward(&x.0)?))
}

/// Mean softmax cross-entropy of `logits` (n, classes) against one label per row.
///
/// This is candle_nn's `cross_entropy`.
pub fn cross_entropy(logits: &Tensor, labels: &[u32]) -> Result<Tensor> {
    let (n, classes) = logits.0.dims2()?;
    let in_range = labels
        .iter()
        .all(|&label| usize::try_from(label).is_ok_and(|label| label < classes));
    if labels.len() != n || !in_range {
        return Err(BrutError::Invalid(format!(
            "cross_entropy: {} labels for {n} rows of {classes} classes",
            labels.len()
        )));
    }

    let targets = candle::Tensor::from_vec(labels.to_vec(), n, logits.0.device())?;

    Ok(Tensor(candle_nn::loss::cross_entropy(&logits.0, &targets)?))
}

/// Mean squared error: the mean of `(prediction - target)^2` over all values.
///
/// Both tensors have the same shape. This is the loss of the MLP for regression
/// (`src/nn.rs`).
pub fn mse(prediction: &Tensor, target: &Tensor) -> Result<Tensor> {
    if prediction.dims() != target.dims() || prediction.0.elem_count() == 0 {
        return Err(BrutError::Invalid(format!(
            "mse: prediction {:?} and target {:?} do not fit",
            prediction.dims(),
            target.dims()
        )));
    }

    Ok(Tensor((&prediction.0 - &target.0)?.sqr()?.mean_all()?))
}

/// The mean over every axis after the channels: (n, c, ...) becomes (n, c).
pub fn global_avg_pool(x: &Tensor) -> Result<Tensor> {
    if x.dims().len() < 3 {
        return Err(BrutError::Invalid(format!(
            "global_avg_pool: the input {:?} has no spatial axis",
            x.dims()
        )));
    }

    Ok(Tensor(x.0.flatten_from(2)?.mean(2)?))
}

/// max(x, 0), elementwise.
pub fn relu(x: &Tensor) -> Result<Tensor> {
    Ok(Tensor(x.0.relu()?))
}

/// min(max(x, 0), 6), elementwise (MobileNetV2).
pub fn relu6(x: &Tensor) -> Result<Tensor> {
    Ok(Tensor(x.0.clamp(0f64, 6f64)?))
}

/// x * sigmoid(x), elementwise (also called swish).
pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(Tensor(x.0.silu()?))
}

/// tanh(x), elementwise.
pub fn tanh(x: &Tensor) -> Result<Tensor> {
    Ok(Tensor(x.0.tanh()?))
}

/// 1 / (1 + exp(-x)), elementwise.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(Tensor(candle_nn::ops::sigmoid(&x.0)?))
}

/// sin(x), elementwise.
pub fn sin(x: &Tensor) -> Result<Tensor> {
    Ok(Tensor(x.0.sin()?))
}

/// GELU in its tanh form, 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))), built from `tanh`.
///
/// The MLP's GELU (`src/nn.rs`) uses the same form, with its own fast tanh.
///
/// Replacement W7 for defect D7. candle's backward pass of `gelu` uses
/// constants rounded to 6 digits (0.0356774, 0.797885, 0.0535161 and 0.398942,
/// candle-core/src/backprop.rs:613-615), so its gradient is off by up to
/// 2.5e-7. The backward pass of `tanh` is exact.
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    let x = &x.0;
    let scale = (2.0 / std::f64::consts::PI).sqrt();
    let cube = (x.sqr()? * x)?;
    let inner = ((x + (cube * 0.044715)?)? * scale)?;

    Ok(Tensor(((x * 0.5)? * (inner.tanh()? + 1.0)?)?))
}

/// GELU in its erf form, 0.5 x (1 + erf(x / sqrt(2))), built from `erf`.
///
/// Replacement W7 for defect D7. candle's backward pass of `gelu_erf` uses
/// 0.398942 for 1 / sqrt(2 pi) (candle-core/src/backprop.rs:628-630), so its
/// gradient is off by up to 1.7e-7. The backward pass of `erf` uses an exact
/// constant.
pub fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    let x = &x.0;
    let scaled = (x * std::f64::consts::FRAC_1_SQRT_2)?;

    Ok(Tensor(((x * 0.5)? * (scaled.erf()? + 1.0)?)?))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// Step of the finite difference.
    const STEP: f64 = 1e-3;

    /// Largest relative error of a gradient entry that passes.
    const LIMIT: f64 = 1e-6;

    /// Entries below this share of the largest numeric gradient are compared against that share.
    const FLOOR: f64 = 1e-3;

    /// Distance of every input from the kinks of piecewise linear operations.
    const KINK_GAP: f64 = 0.01;

    /// Largest difference from candle's own forward pass, in f64.
    const FORWARD_LIMIT: f64 = 1e-12;

    /// Seed of the weights r in the checked scalar sum(f(x) * r).
    const WEIGHT_SEED: u64 = 20261031;

    /// An operation under test, applied to the inputs of a case.
    type Op = Box<dyn Fn(&[Tensor]) -> Result<Tensor>>;

    /// One input of a case: its shape and its values.
    struct Input {
        shape: Vec<usize>,
        values: Vec<f64>,
    }

    /// One gradient check of the public function `op`.
    struct Case {
        op: &'static str,
        label: &'static str,
        inputs: Vec<Input>,
        f: Op,
    }

    /// Collects cases and draws their inputs from one seeded ChaCha8 stream.
    struct Cases {
        rng: ChaCha8Rng,
        list: Vec<Case>,
    }

    impl Cases {
        fn new(seed: u64) -> Cases {
            Cases {
                rng: ChaCha8Rng::seed_from_u64(seed),
                list: Vec::new(),
            }
        }

        /// Values uniform in [lo, hi).
        fn uniform(&mut self, shape: &[usize], lo: f64, hi: f64) -> Input {
            let count = shape.iter().product();
            let values = (0..count).map(|_| self.rng.random_range(lo..hi)).collect();

            Input {
                shape: shape.to_vec(),
                values,
            }
        }

        /// Values uniform in [-1, 1).
        fn plain(&mut self, shape: &[usize]) -> Input {
            self.uniform(shape, -1.0, 1.0)
        }

        /// Values in [lo, hi) that keep KINK_GAP away from every kink.
        fn kinked(&mut self, shape: &[usize], lo: f64, hi: f64, kinks: &[f64]) -> Input {
            let count = shape.iter().product();
            let mut values = Vec::with_capacity(count);
            while values.len() < count {
                let v = self.rng.random_range(lo..hi);
                if kinks.iter().all(|k| (v - k).abs() >= KINK_GAP) {
                    values.push(v);
                }
            }

            Input {
                shape: shape.to_vec(),
                values,
            }
        }

        /// Distinct values KINK_GAP apart, in random order, so no max window has a near tie.
        fn distinct(&mut self, shape: &[usize]) -> Input {
            let count: usize = shape.iter().product();
            let mut order: Vec<usize> = (0..count).collect();
            for i in (1..count).rev() {
                let j = self.rng.random_range(0..=i);
                order.swap(i, j);
            }

            let centre = count as f64 / 2.0;
            let values = order
                .into_iter()
                .map(|k| (k as f64 - centre) * KINK_GAP)
                .collect();

            Input {
                shape: shape.to_vec(),
                values,
            }
        }

        fn add(
            &mut self,
            op: &'static str,
            label: &'static str,
            inputs: Vec<Input>,
            f: impl Fn(&[Tensor]) -> Result<Tensor> + 'static,
        ) {
            self.list.push(Case {
                op,
                label,
                inputs,
                f: Box::new(f),
            });
        }
    }

    /// A plain (not trainable) f64 tensor.
    fn f64_tensor(values: Vec<f64>, shape: &[usize]) -> Tensor {
        Tensor(candle::Tensor::from_vec(values, shape, &Device::Cpu).expect("an f64 tensor"))
    }

    /// The input as a plain f64 tensor.
    fn tensor_of(input: &Input) -> Tensor {
        f64_tensor(input.values.clone(), &input.shape)
    }

    /// All values of an f64 tensor, in row-major order.
    fn f64_values(t: &Tensor) -> Vec<f64> {
        t.0.flatten_all()
            .and_then(|flat| flat.to_vec1::<f64>())
            .expect("f64 values")
    }

    /// The checked scalar, sum(f(x) * r).
    fn weighted_sum(out: &Tensor, r: &candle::Tensor) -> Result<Tensor> {
        Ok(Tensor((&out.0 * r)?.sum_all()?))
    }

    /// Four-point central difference of the checked scalar for every entry of input `k`.
    fn numeric_grad(case: &Case, k: usize, r: &candle::Tensor) -> Result<Vec<f64>> {
        let loss_at = |values: &[Vec<f64>]| -> Result<f64> {
            let xs: Vec<Tensor> = values
                .iter()
                .zip(&case.inputs)
                .map(|(v, input)| f64_tensor(v.clone(), &input.shape))
                .collect();
            let out = (case.f)(&xs)?;

            Ok(weighted_sum(&out, r)?.0.to_scalar::<f64>()?)
        };

        let mut values: Vec<Vec<f64>> = case.inputs.iter().map(|i| i.values.clone()).collect();
        let mut grad = Vec::with_capacity(values[k].len());
        for j in 0..values[k].len() {
            let base = values[k][j];
            let mut losses = [0.0; 4];
            for (loss, delta) in losses
                .iter_mut()
                .zip([2.0 * STEP, STEP, -STEP, -2.0 * STEP])
            {
                values[k][j] = base + delta;
                *loss = loss_at(&values)?;
            }
            values[k][j] = base;

            let [p2, p1, m1, m2] = losses;
            grad.push((-p2 + 8.0 * p1 - 8.0 * m1 + m2) / (12.0 * STEP));
        }

        Ok(grad)
    }

    /// Largest |a - n| / max(|a|, |n|, FLOOR * largest |n|) over the entries.
    fn max_rel_error(analytic: &[f64], numeric: &[f64]) -> f64 {
        if analytic.len() != numeric.len() {
            return f64::INFINITY;
        }

        let scale = numeric.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let floor = (FLOOR * scale).max(f64::MIN_POSITIVE);
        let mut worst = 0.0f64;
        for (a, n) in analytic.iter().zip(numeric) {
            if !a.is_finite() || !n.is_finite() {
                return f64::INFINITY;
            }

            worst = worst.max((a - n).abs() / a.abs().max(n.abs()).max(floor));
        }

        worst
    }

    /// Checks one case; returns its largest relative error, or what failed.
    fn check(case: &Case) -> std::result::Result<f64, String> {
        let fail = |what: String| format!("{} ({}): {what}", case.op, case.label);

        let params = case
            .inputs
            .iter()
            .map(|input| Param::new(&tensor_of(input)))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| fail(e.to_string()))?;
        let xs: Vec<Tensor> = params.iter().map(Param::tensor).collect();
        let out = (case.f)(&xs).map_err(|e| fail(format!("forward: {e}")))?;

        let mut rng = ChaCha8Rng::seed_from_u64(WEIGHT_SEED);
        let weights: Vec<f64> = (0..out.0.elem_count())
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let r = candle::Tensor::from_vec(weights, out.dims(), &Device::Cpu)
            .map_err(|e| fail(e.to_string()))?;
        let grads = weighted_sum(&out, &r)
            .and_then(|loss| backward(&loss))
            .map_err(|e| fail(format!("backward: {e}")))?;

        let mut worst = 0.0f64;
        for (k, param) in params.iter().enumerate() {
            let analytic = grads
                .get(param)
                .ok_or_else(|| fail(format!("no gradient reaches input {k}")))?;
            let numeric = numeric_grad(case, k, &r).map_err(|e| fail(e.to_string()))?;
            let error = max_rel_error(&f64_values(&analytic), &numeric);
            if error.is_nan() || error >= LIMIT {
                return Err(fail(format!("input {k}: relative error {error:.2e}")));
            }

            worst = worst.max(error);
        }

        Ok(worst)
    }

    /// Checks every case and fails with the list of failures.
    fn run(cases: &[Case]) {
        let mut failures = Vec::new();
        for case in cases {
            match check(case) {
                Ok(worst) => eprintln!("gradient ok: {} ({}): {worst:.1e}", case.op, case.label),
                Err(message) => failures.push(message),
            }
        }

        assert!(
            failures.is_empty(),
            "gradient check failed:\n{}",
            failures.join("\n")
        );
    }

    /// Running mean 0 and running variance 1 for `c` channels, in f64.
    fn fresh_stats(c: usize) -> (Tensor, Tensor) {
        (
            f64_tensor(vec![0.0; c], &[c]),
            f64_tensor(vec![1.0; c], &[c]),
        )
    }

    /// Convolutions, at the shapes where candle's own versions fail.
    fn conv_cases() -> Vec<Case> {
        let mut c = Cases::new(1);

        let inputs = vec![c.plain(&[2, 3, 5, 6]), c.plain(&[4, 3, 3, 3])];
        c.add("conv2d", "3x3, padding 1, 5x6", inputs, |t| {
            conv2d(&t[0], &t[1], 1, 1, 1, 1)
        });

        let inputs = vec![c.plain(&[2, 3, 7, 8]), c.plain(&[4, 3, 3, 3])];
        c.add("conv2d", "3x3, stride 2, 7x8, crop path", inputs, |t| {
            conv2d(&t[0], &t[1], 1, 2, 1, 1)
        });

        let inputs = vec![c.plain(&[2, 2, 9, 10]), c.plain(&[3, 2, 3, 3])];
        c.add("conv2d", "3x3, stride 2, dilation 2, 9x10", inputs, |t| {
            conv2d(&t[0], &t[1], 1, 2, 2, 1)
        });

        let inputs = vec![c.plain(&[2, 3, 7, 7]), c.plain(&[4, 3, 3, 3])];
        c.add("conv2d", "3x3, stride 2, 7x7, native path", inputs, |t| {
            conv2d(&t[0], &t[1], 1, 2, 1, 1)
        });

        let inputs = vec![c.plain(&[2, 3, 6, 8]), c.plain(&[4, 3, 3, 3])];
        c.add(
            "conv2d",
            "3x3, stride 2, 6x8, equal remainders",
            inputs,
            |t| conv2d(&t[0], &t[1], 1, 2, 1, 1),
        );

        let inputs = vec![c.plain(&[2, 4, 7, 8]), c.plain(&[6, 2, 3, 3])];
        c.add("conv2d", "groups 2, stride 2, 7x8", inputs, |t| {
            conv2d(&t[0], &t[1], 1, 2, 1, 2)
        });

        let inputs = vec![c.plain(&[2, 4, 6, 7]), c.plain(&[4, 1, 3, 3])];
        c.add("conv2d", "depthwise 3x3, 6x7", inputs, |t| {
            conv2d(&t[0], &t[1], 1, 1, 1, 4)
        });

        let inputs = vec![c.plain(&[2, 3, 7, 8]), c.plain(&[3, 1, 3, 3])];
        c.add("conv2d", "depthwise 3x3, stride 2, 7x8", inputs, |t| {
            conv2d(&t[0], &t[1], 1, 2, 1, 3)
        });

        let inputs = vec![c.plain(&[1, 3, 8, 9]), c.plain(&[3, 1, 7, 7])];
        c.add("conv2d", "depthwise 7x7, padding 3, 8x9", inputs, |t| {
            conv2d(&t[0], &t[1], 3, 1, 1, 3)
        });

        let inputs = vec![c.plain(&[2, 3, 5, 6]), c.plain(&[4, 3, 1, 1])];
        c.add("conv2d", "1x1, stride 2, 5x6", inputs, |t| {
            conv2d(&t[0], &t[1], 0, 2, 1, 1)
        });

        let inputs = vec![c.plain(&[1, 2, 9, 8]), c.plain(&[3, 2, 5, 5])];
        c.add("conv2d", "5x5, stride 2, padding 2, 9x8", inputs, |t| {
            conv2d(&t[0], &t[1], 2, 2, 1, 1)
        });

        let inputs = vec![c.plain(&[2, 3, 9]), c.plain(&[4, 3, 3])];
        c.add("conv1d", "k3, padding 1, length 9", inputs, |t| {
            conv1d(&t[0], &t[1], 1, 1, 1, 1)
        });

        let inputs = vec![c.plain(&[2, 3, 10]), c.plain(&[4, 3, 3])];
        c.add("conv1d", "k3, stride 2, length 10", inputs, |t| {
            conv1d(&t[0], &t[1], 1, 2, 1, 1)
        });

        let inputs = vec![c.plain(&[2, 3, 9]), c.plain(&[4, 3, 3])];
        c.add("conv1d", "k3, dilation 2, length 9", inputs, |t| {
            conv1d(&t[0], &t[1], 2, 1, 2, 1)
        });

        let inputs = vec![c.plain(&[2, 4, 9]), c.plain(&[4, 2, 3])];
        c.add("conv1d", "k3, groups 2, length 9", inputs, |t| {
            conv1d(&t[0], &t[1], 1, 1, 1, 2)
        });

        let inputs = vec![c.plain(&[2, 4, 9]), c.plain(&[4, 1, 3])];
        c.add("conv1d", "k3, depthwise, length 9", inputs, |t| {
            conv1d(&t[0], &t[1], 1, 1, 1, 4)
        });

        let inputs = vec![c.plain(&[2, 4, 11]), c.plain(&[4, 2, 3])];
        c.add(
            "conv1d",
            "k3, stride 2, dilation 2, groups 2",
            inputs,
            |t| conv1d(&t[0], &t[1], 1, 2, 2, 2),
        );

        let inputs = vec![c.plain(&[2, 3, 9]), c.plain(&[4, 3, 3])];
        c.add("causal_conv1d", "k3, dilation 2, length 9", inputs, |t| {
            causal_conv1d(&t[0], &t[1], 2, 1)
        });

        let inputs = vec![c.plain(&[3, 4, 10]), c.plain(&[4, 2, 2])];
        c.add("causal_conv1d", "k2, dilation 4, groups 2", inputs, |t| {
            causal_conv1d(&t[0], &t[1], 4, 2)
        });

        let inputs = vec![c.plain(&[2, 3, 3, 4]), c.plain(&[3, 4, 2, 2])];
        c.add("conv_transpose2d", "2x2, stride 2, 3x4", inputs, |t| {
            conv_transpose2d(&t[0], &t[1], 0, 0, 2, 1)
        });

        let inputs = vec![c.plain(&[2, 3, 3, 4]), c.plain(&[3, 2, 3, 3])];
        c.add(
            "conv_transpose2d",
            "3x3, stride 2, padding 1, output padding 1, 3x4",
            inputs,
            |t| conv_transpose2d(&t[0], &t[1], 1, 1, 2, 1),
        );

        let inputs = vec![c.plain(&[1, 2, 4, 5]), c.plain(&[2, 3, 3, 3])];
        c.add("conv_transpose2d", "3x3, dilation 2, 4x5", inputs, |t| {
            conv_transpose2d(&t[0], &t[1], 0, 0, 1, 2)
        });

        c.list
    }

    /// Pooling and upsampling, with odd sizes and kernels unequal to the stride.
    fn pool_cases() -> Vec<Case> {
        let mut c = Cases::new(2);

        let inputs = vec![c.distinct(&[2, 3, 7, 7])];
        c.add("max_pool2d", "2x2, stride 2, 7x7", inputs, |t| {
            max_pool2d(&t[0], 2, 2, 0)
        });

        let inputs = vec![c.distinct(&[2, 2, 6, 8])];
        c.add("max_pool2d", "2x2, stride 2, 6x8", inputs, |t| {
            max_pool2d(&t[0], 2, 2, 0)
        });

        let inputs = vec![c.distinct(&[2, 2, 7, 8])];
        c.add("max_pool2d", "3x3, stride 2, padding 1, 7x8", inputs, |t| {
            max_pool2d(&t[0], 3, 2, 1)
        });

        let inputs = vec![c.distinct(&[1, 3, 5, 6])];
        c.add("max_pool2d", "3x3, stride 1, padding 1, 5x6", inputs, |t| {
            max_pool2d(&t[0], 3, 1, 1)
        });

        let inputs = vec![c.plain(&[2, 3, 7, 9])];
        c.add("avg_pool2d", "2x2, stride 2, 7x9, crop path", inputs, |t| {
            avg_pool2d(&t[0], 2, 2, 0)
        });

        let inputs = vec![c.plain(&[1, 2, 8, 7])];
        c.add("avg_pool2d", "3x3, stride 3, 8x7, crop path", inputs, |t| {
            avg_pool2d(&t[0], 3, 3, 0)
        });

        let inputs = vec![c.plain(&[2, 3, 5, 6])];
        c.add("avg_pool2d", "3x3, stride 1, padding 1, 5x6", inputs, |t| {
            avg_pool2d(&t[0], 3, 1, 1)
        });

        let inputs = vec![c.plain(&[2, 2, 7, 8])];
        c.add("avg_pool2d", "3x3, stride 2, padding 1, 7x8", inputs, |t| {
            avg_pool2d(&t[0], 3, 2, 1)
        });

        let inputs = vec![c.distinct(&[2, 3, 9])];
        c.add("max_pool1d", "window 2, stride 2, length 9", inputs, |t| {
            max_pool1d(&t[0], 2, 2)
        });

        let inputs = vec![c.distinct(&[2, 2, 10])];
        c.add("max_pool1d", "window 3, stride 2, length 10", inputs, |t| {
            max_pool1d(&t[0], 3, 2)
        });

        let inputs = vec![c.plain(&[2, 3, 10])];
        c.add("avg_pool1d", "window 4, stride 4, length 10", inputs, |t| {
            avg_pool1d(&t[0], 4, 4)
        });

        let inputs = vec![c.plain(&[2, 2, 9])];
        c.add("avg_pool1d", "window 3, stride 2, length 9", inputs, |t| {
            avg_pool1d(&t[0], 3, 2)
        });

        let inputs = vec![c.plain(&[2, 3, 3, 4])];
        c.add("upsample_nearest2d", "factor 2, 3x4", inputs, |t| {
            upsample_nearest2d(&t[0], 2)
        });

        let inputs = vec![c.plain(&[1, 2, 2, 3])];
        c.add("upsample_nearest2d", "factor 3, 2x3", inputs, |t| {
            upsample_nearest2d(&t[0], 3)
        });

        // The input also feeds x * x; candle's own upsampling loses that gradient.
        let inputs = vec![c.plain(&[2, 3, 3, 3])];
        c.add(
            "upsample_nearest2d",
            "input shared with x * x, upsampling first",
            inputs,
            |t| {
                let up = flatten(&upsample_nearest2d(&t[0], 2)?)?;
                let square = flatten(&broadcast_mul(&t[0], &t[0])?)?;
                cat(&[&up, &square], 1)
            },
        );

        let inputs = vec![c.plain(&[2, 3, 3, 3])];
        c.add(
            "upsample_nearest2d",
            "input shared with x * x, product first",
            inputs,
            |t| {
                let up = flatten(&upsample_nearest2d(&t[0], 2)?)?;
                let square = flatten(&broadcast_mul(&t[0], &t[0])?)?;
                cat(&[&square, &up], 1)
            },
        );

        let inputs = vec![c.plain(&[2, 3, 4, 5])];
        c.add("global_avg_pool", "2x3x4x5", inputs, |t| {
            global_avg_pool(&t[0])
        });

        let inputs = vec![c.plain(&[2, 3, 7])];
        c.add("global_avg_pool", "sequence 2x3x7", inputs, |t| {
            global_avg_pool(&t[0])
        });

        c.list
    }

    /// Batch, layer and group norm.
    fn norm_cases() -> Vec<Case> {
        let mut c = Cases::new(3);

        let inputs = vec![
            c.plain(&[4, 3, 3, 3]),
            c.uniform(&[3], 0.5, 1.5),
            c.plain(&[3]),
        ];
        c.add("forward_train", "batch norm, 4x3x3x3", inputs, |t| {
            let (mean, var) = fresh_stats(3);
            BatchNorm::new(&t[1], &t[2], &mean, &var, 1e-5)?.forward_train(&t[0])
        });

        let inputs = vec![
            c.plain(&[3, 2, 5]),
            c.uniform(&[2], 0.5, 1.5),
            c.plain(&[2]),
        ];
        c.add("forward_train", "batch norm, sequence 3x2x5", inputs, |t| {
            let (mean, var) = fresh_stats(2);
            BatchNorm::new(&t[1], &t[2], &mean, &var, 1e-5)?.forward_train(&t[0])
        });

        let inputs = vec![
            c.plain(&[2, 4, 3, 3]),
            c.uniform(&[4], 0.5, 1.5),
            c.plain(&[4]),
        ];
        c.add("layer_norm_channels", "2x4x3x3", inputs, |t| {
            layer_norm_channels(&t[0], &t[1], &t[2], 1e-5)
        });

        let inputs = vec![
            c.plain(&[2, 4, 5]),
            c.uniform(&[4], 0.5, 1.5),
            c.plain(&[4]),
        ];
        c.add("layer_norm_channels", "sequence 2x4x5", inputs, |t| {
            layer_norm_channels(&t[0], &t[1], &t[2], 1e-5)
        });

        let inputs = vec![
            c.plain(&[2, 4, 3, 3]),
            c.uniform(&[4], 0.5, 1.5),
            c.plain(&[4]),
        ];
        c.add("group_norm", "2 groups of 2 channels", inputs, |t| {
            group_norm(&t[0], &t[1], &t[2], 2, 1e-5)
        });

        let inputs = vec![
            c.plain(&[2, 3, 5]),
            c.uniform(&[3], 0.5, 1.5),
            c.plain(&[3]),
        ];
        c.add("group_norm", "1 group, sequence 2x3x5", inputs, |t| {
            group_norm(&t[0], &t[1], &t[2], 1, 1e-5)
        });

        let inputs = vec![
            c.plain(&[2, 4, 3, 3]),
            c.uniform(&[4], 0.5, 1.5),
            c.plain(&[4]),
        ];
        c.add("group_norm", "one group per channel", inputs, |t| {
            group_norm(&t[0], &t[1], &t[2], 4, 1e-5)
        });

        c.list
    }

    /// Joins, dense layers, the loss, and inputs that feed two branches.
    fn structure_cases() -> Vec<Case> {
        let mut c = Cases::new(4);

        let inputs = vec![c.plain(&[2, 3, 4, 4]), c.plain(&[2, 2, 4, 4])];
        c.add("cat", "along the channels", inputs, |t| {
            cat(&[&t[0], &t[1]], 1)
        });

        let inputs = vec![
            c.plain(&[2, 3, 2]),
            c.plain(&[2, 3, 3]),
            c.plain(&[2, 3, 1]),
        ];
        c.add("cat", "three parts along the last axis", inputs, |t| {
            cat(&[&t[0], &t[1], &t[2]], 2)
        });

        let inputs = vec![c.plain(&[2, 4, 3, 3]), c.plain(&[2, 4, 1, 1])];
        c.add(
            "broadcast_mul",
            "(2, 4, 3, 3) times (2, 4, 1, 1)",
            inputs,
            |t| broadcast_mul(&t[0], &t[1]),
        );

        // Squeeze-and-excitation: x feeds the gate and the product. b1 keeps the ReLU inputs positive.
        let inputs = vec![
            c.plain(&[2, 4, 3, 3]),
            c.uniform(&[2, 4], -0.2, 0.2),
            c.uniform(&[2], 1.0, 1.5),
            c.plain(&[4, 2]),
            c.plain(&[4]),
        ];
        c.add(
            "broadcast_mul",
            "squeeze-and-excitation, input shared",
            inputs,
            |t| {
                let squeezed = global_avg_pool(&t[0])?;
                let hidden = relu(&linear(&squeezed, &t[1], &t[2])?)?;
                let gate = sigmoid(&linear(&hidden, &t[3], &t[4])?)?;
                broadcast_mul(&t[0], &reshape(&gate, &[2, 4, 1, 1])?)
            },
        );

        let inputs = vec![c.plain(&[2, 4, 3, 3]), c.plain(&[1, 4, 1, 1])];
        c.add(
            "broadcast_add",
            "(2, 4, 3, 3) plus a bias (1, 4, 1, 1)",
            inputs,
            |t| broadcast_add(&t[0], &t[1]),
        );

        // A biased 1-D convolution: x feeds the convolution, the bias every position.
        let inputs = vec![c.plain(&[2, 3, 8]), c.plain(&[4, 3, 3]), c.plain(&[1, 4, 1])];
        c.add("broadcast_add", "conv1d plus a bias (1, 4, 1)", inputs, |t| {
            broadcast_add(&conv1d(&t[0], &t[1], 1, 1, 1, 1)?, &t[2])
        });

        let inputs = vec![c.plain(&[2, 4, 5, 5]), c.plain(&[4, 4, 3, 3])];
        c.add("add", "residual x + conv2d(x), input shared", inputs, |t| {
            add(&t[0], &conv2d(&t[0], &t[1], 1, 1, 1, 1)?)
        });

        let inputs = vec![c.plain(&[2, 6, 3, 3])];
        c.add("channel_shuffle", "6 channels in 2 groups", inputs, |t| {
            channel_shuffle(&t[0], 2)
        });

        let inputs = vec![c.plain(&[2, 6, 3, 3])];
        c.add("channel_shuffle", "6 channels in 3 groups", inputs, |t| {
            channel_shuffle(&t[0], 3)
        });

        let inputs = vec![c.plain(&[2, 12])];
        c.add("reshape", "(2, 12) to (2, 3, 2, 2)", inputs, |t| {
            reshape(&t[0], &[2, 3, 2, 2])
        });

        let inputs = vec![c.plain(&[3, 5]), c.plain(&[4, 5]), c.plain(&[4])];
        c.add("linear", "3x5 times 4x5, plus 4", inputs, |t| {
            linear(&t[0], &t[1], &t[2])
        });

        let inputs = vec![c.plain(&[2, 3, 2, 2]), c.plain(&[4, 12]), c.plain(&[4])];
        c.add("flatten", "2x3x2x2, then a dense layer", inputs, |t| {
            linear(&flatten(&t[0])?, &t[1], &t[2])
        });

        let inputs = vec![c.uniform(&[5, 4], -2.0, 2.0)];
        c.add("cross_entropy", "5 rows of 4 classes", inputs, |t| {
            cross_entropy(&t[0], &[0, 3, 1, 2, 1])
        });

        let inputs = vec![c.plain(&[5, 3]), c.plain(&[5, 3])];
        c.add("mse", "5 rows of 3 values", inputs, |t| mse(&t[0], &t[1]));

        let inputs = vec![c.plain(&[4, 6]), c.plain(&[2, 6]), c.plain(&[2]), c.plain(&[4, 2])];
        c.add("mse", "a dense layer against fixed targets", inputs, |t| {
            mse(&linear(&t[0], &t[1], &t[2])?, &t[3])
        });

        // A small smooth network, end to end: every input reaches the loss through several ops.
        let inputs = vec![
            c.plain(&[4, 1, 7, 8]),
            c.plain(&[3, 1, 3, 3]),
            c.uniform(&[3], 0.5, 1.5),
            c.plain(&[3]),
            c.plain(&[4, 3, 3, 3]),
            c.uniform(&[4], 0.5, 1.5),
            c.plain(&[4]),
            c.plain(&[2, 4]),
            c.plain(&[2]),
        ];
        c.add("cross_entropy", "small CNN, end to end", inputs, |t| {
            let (mean, var) = fresh_stats(3);
            let stem = conv2d(&t[0], &t[1], 1, 1, 1, 1)?;
            let stem = BatchNorm::new(&t[2], &t[3], &mean, &var, 1e-5)?.forward_train(&stem)?;
            let pooled = avg_pool2d(&silu(&stem)?, 2, 2, 0)?;
            let down = conv2d(&pooled, &t[4], 1, 2, 1, 1)?;
            let down = gelu_tanh(&layer_norm_channels(&down, &t[5], &t[6], 1e-5)?)?;
            let logits = linear(&global_avg_pool(&down)?, &t[7], &t[8])?;
            cross_entropy(&logits, &[0, 1, 1, 0])
        });

        c.list
    }

    /// Elementwise activations; inputs of piecewise linear ones keep away from their kinks.
    fn activation_cases() -> Vec<Case> {
        let mut c = Cases::new(5);
        let shape = [2, 3, 4, 4];

        let inputs = vec![c.kinked(&shape, -2.0, 2.0, &[0.0])];
        c.add("relu", "kinks avoided", inputs, |t| relu(&t[0]));

        let inputs = vec![c.kinked(&shape, -3.0, 9.0, &[0.0, 6.0])];
        c.add("relu6", "kinks avoided", inputs, |t| relu6(&t[0]));

        let inputs = vec![c.uniform(&shape, -3.0, 3.0)];
        c.add("silu", "-3 to 3", inputs, |t| silu(&t[0]));

        let inputs = vec![c.uniform(&shape, -3.0, 3.0)];
        c.add("tanh", "-3 to 3", inputs, |t| tanh(&t[0]));

        let inputs = vec![c.uniform(&shape, -4.0, 4.0)];
        c.add("sigmoid", "-4 to 4", inputs, |t| sigmoid(&t[0]));

        let inputs = vec![c.uniform(&shape, -3.0, 3.0)];
        c.add("sin", "-3 to 3", inputs, |t| sin(&t[0]));

        let inputs = vec![c.uniform(&shape, -3.0, 3.0)];
        c.add("gelu_tanh", "-3 to 3", inputs, |t| gelu_tanh(&t[0]));

        let inputs = vec![c.uniform(&shape, -3.0, 3.0)];
        c.add("gelu_erf", "-3 to 3", inputs, |t| gelu_erf(&t[0]));

        c.list
    }

    #[test]
    fn convolutions_have_correct_gradients() {
        run(&conv_cases());
    }

    #[test]
    fn pooling_and_upsampling_have_correct_gradients() {
        run(&pool_cases());
    }

    #[test]
    fn norms_have_correct_gradients() {
        run(&norm_cases());
    }

    #[test]
    fn joins_dense_layers_and_the_loss_have_correct_gradients() {
        run(&structure_cases());
    }

    #[test]
    fn activations_have_correct_gradients() {
        run(&activation_cases());
    }

    /// candle's own operations fail the same check where a replacement exists (D1 to D8).
    ///
    /// This shows that the check can fail. If a later candle commit fixes a
    /// defect, this test fails, and the replacement may go (ADR 0007).
    #[test]
    fn candle_defects_fail_the_gradient_check() {
        let mut c = Cases::new(6);

        let inputs = vec![c.plain(&[2, 3, 9]), c.plain(&[4, 3, 3])];
        c.add("conv1d", "D1, candle's conv1d", inputs, |t| {
            Ok(Tensor(t[0].0.conv1d(&t[1].0, 1, 1, 1, 1)?))
        });

        let inputs = vec![c.plain(&[2, 3, 7, 8]), c.plain(&[4, 3, 3, 3])];
        c.add(
            "conv2d",
            "D2, candle's conv2d, stride 2 on 7x8",
            inputs,
            |t| Ok(Tensor(t[0].0.conv2d(&t[1].0, 1, 2, 1, 1)?)),
        );

        let inputs = vec![c.distinct(&[1, 2, 4, 4])];
        c.add("max_pool2d", "D3, candle's max_pool2d", inputs, |t| {
            Ok(Tensor(t[0].0.max_pool2d(2)?))
        });

        let inputs = vec![c.plain(&[2, 3, 7, 7])];
        c.add(
            "avg_pool2d",
            "D4, candle's avg_pool2d on 7x7",
            inputs,
            |t| Ok(Tensor(t[0].0.avg_pool2d(2)?)),
        );

        let inputs = vec![c.plain(&[2, 3, 3, 3])];
        c.add(
            "upsample_nearest2d",
            "D5, candle's upsample_nearest2d, input shared",
            inputs,
            |t| {
                let up = Tensor(t[0].0.upsample_nearest2d(6, 6)?);
                let square = broadcast_mul(&t[0], &t[0])?;
                cat(&[&flatten(&up)?, &flatten(&square)?], 1)
            },
        );

        let inputs = vec![
            c.plain(&[2, 4, 3, 3]),
            c.uniform(&[4], 0.5, 1.5),
            c.plain(&[4]),
        ];
        c.add(
            "layer_norm_channels",
            "D6, candle_nn's LayerNorm",
            inputs,
            |t| {
                let nhwc = t[0].0.permute((0, 2, 3, 1))?.contiguous()?;
                let norm = candle_nn::LayerNorm::new(t[1].0.clone(), t[2].0.clone(), 1e-5);
                Ok(Tensor(norm.forward(&nhwc)?.permute((0, 3, 1, 2))?))
            },
        );

        let inputs = vec![c.uniform(&[2, 3, 4, 4], -3.0, 3.0)];
        c.add("gelu_tanh", "D7, candle's gelu", inputs, |t| {
            Ok(Tensor(t[0].0.gelu()?))
        });

        let inputs = vec![c.uniform(&[2, 3, 4, 4], -3.0, 3.0)];
        c.add("gelu_erf", "D7, candle's gelu_erf", inputs, |t| {
            Ok(Tensor(t[0].0.gelu_erf()?))
        });

        for case in &c.list {
            let outcome = check(case);
            eprintln!("candle's own op: {} ({}): {outcome:?}", case.op, case.label);
            assert!(outcome.is_err(), "{} ({}) passed", case.op, case.label);
        }

        // D8: candle cannot seed its CPU generator.
        assert!(Device::Cpu.set_seed(1).is_err());
    }

    /// Public functions of this file that are not differentiable operations.
    const NOT_OPS: [&str; 13] = [
        "from_f32",
        "normal",
        "dims",
        "to_vec_f32",
        "new",
        "tensor",
        "set",
        "get",
        "backward",
        "out_len",
        "forward_eval",
        "running_mean",
        "running_var",
    ];

    /// This file without its test module.
    fn production_source() -> &'static str {
        let source = include_str!("ops.rs");
        let end = source
            .find("\n#[cfg(test)]\nmod tests")
            .expect("ops.rs has a test module");

        &source[..end]
    }

    /// The code of a line, without a `//` comment.
    fn code_of(line: &str) -> &str {
        match line.find("//") {
            Some(start) => &line[..start],
            None => line,
        }
    }

    /// The name of the function whose signature starts on this line, if any.
    fn opened_fn(code: &str) -> Option<&str> {
        let code = code.trim_start();
        let code = code.strip_prefix("pub ").unwrap_or(code);
        let rest = code.strip_prefix("fn ")?;
        let end = rest
            .find(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
            .unwrap_or(rest.len());

        Some(&rest[..end])
    }

    /// The names of the public functions in `source`.
    fn public_fns(source: &str) -> BTreeSet<&str> {
        source
            .lines()
            .map(code_of)
            .filter(|code| code.trim_start().starts_with("pub fn "))
            .filter_map(opened_fn)
            .collect()
    }

    #[test]
    fn every_public_op_has_a_gradient_case() {
        let public = public_fns(production_source());
        let groups = [
            conv_cases(),
            pool_cases(),
            norm_cases(),
            structure_cases(),
            activation_cases(),
        ];
        let checked: BTreeSet<&str> = groups.iter().flatten().map(|case| case.op).collect();

        for name in &public {
            if !NOT_OPS.contains(name) {
                assert!(checked.contains(name), "{name} has no gradient case");
            }
        }

        for name in &checked {
            assert!(
                public.contains(name),
                "a case checks {name}, which is not public"
            );
        }

        assert!(
            public.len() > 30,
            "only {} public functions found",
            public.len()
        );
    }

    /// candle calls with a known defect (ADR 0007), and the functions of this file allowed to make them.
    const KNOWN_WRONG: [(&str, &[&str]); 30] = [
        // D1: the kernel gradient of conv1d.
        (".conv1d(", &[]),
        ("::conv1d(", &[]),
        ("conv1d_with_algo", &[]),
        ("Conv1d", &[]),
        // D2: candle's conv2d only inside conv2d, which takes the crop path where needed.
        (".conv2d(", &["conv2d"]),
        ("::conv2d(", &[]),
        ("conv2d_with_algo", &[]),
        ("Conv2d", &[]),
        // D3: max pooling.
        (".max_pool", &[]),
        ("::max_pool", &[]),
        // D4: average pooling, only in the crop path of W4.
        (".avg_pool", &["avg_pool2d_even"]),
        ("::avg_pool", &[]),
        // D5: nearest upsampling, also behind interpolate.
        (".upsample_nearest", &[]),
        ("::upsample_nearest", &[]),
        ("interpolate", &[]),
        // D6: the fused layer norm and its relatives.
        ("LayerNorm", &[]),
        ("layer_norm(", &[]),
        ("RmsNorm", &[]),
        ("rms_norm", &[]),
        // D7: GELU, also behind candle_nn's activations.
        (".gelu", &[]),
        ("::gelu", &[]),
        ("Activation", &[]),
        // D8: candle's random numbers.
        ("::rand(", &[]),
        ("::randn(", &[]),
        ("rand_f64", &[]),
        ("randn_f64", &[]),
        ("rand_like", &[]),
        ("randn_like", &[]),
        ("set_seed", &[]),
        ("VarBuilder", &[]),
    ];

    /// Lines of `source` that make a known-wrong candle call outside its allowed functions.
    fn known_wrong_calls(source: &str) -> Vec<String> {
        let mut current = "";
        let mut found = Vec::new();
        for (number, line) in source.lines().enumerate() {
            let code = code_of(line);
            if let Some(name) = opened_fn(code) {
                current = name;
            }

            for (pattern, allowed) in KNOWN_WRONG {
                if code.contains(pattern) && !allowed.contains(&current) {
                    found.push(format!("ops.rs:{}: {pattern} in {current}", number + 1));
                }
            }
        }

        found
    }

    #[test]
    fn ops_makes_no_known_wrong_candle_call() {
        assert_eq!(known_wrong_calls(production_source()), Vec::<String>::new());

        // The scan finds such calls, and allows avg_pool2d only in the crop path of W4.
        let planted =
            "fn pool(x: &Tensor) -> Result<Tensor> {\n    Ok(Tensor(x.0.max_pool2d(2)?))\n}\n";
        assert_eq!(known_wrong_calls(planted).len(), 1);

        let planted = "fn other(x: &candle::Tensor) {\n    x.avg_pool2d(2)\n}\n";
        assert_eq!(known_wrong_calls(planted).len(), 1);

        let allowed = "fn avg_pool2d_even(x: &candle::Tensor) {\n    x.avg_pool2d(2)\n}\n";
        assert!(known_wrong_calls(allowed).is_empty());

        let commented = "fn f() {\n    // candle's x.gelu() is wrong\n}\n";
        assert!(known_wrong_calls(commented).is_empty());
    }

    /// Every `.rs` file below `dir`.
    fn rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir).expect("a readable source directory");
        for entry in entries {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                rust_files(&path, files);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
    }

    /// Lines of `source` that name the candle crates.
    fn candle_mentions(source: &str) -> Vec<usize> {
        source
            .lines()
            .enumerate()
            .filter(|(_, line)| {
                let code = code_of(line);
                ["candle_core", "candle_nn", "candle::"]
                    .iter()
                    .any(|name| code.contains(name))
            })
            .map(|(number, _)| number + 1)
            .collect()
    }

    #[test]
    fn no_other_file_calls_candle() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let ops = src.join("cnn").join("ops.rs");
        let mut files = Vec::new();
        rust_files(&src, &mut files);
        files.sort();

        let mut found = Vec::new();
        for file in files.iter().filter(|file| **file != ops) {
            let source = std::fs::read_to_string(file).expect("a readable source file");
            for number in candle_mentions(&source) {
                found.push(format!("{}:{number}", file.display()));
            }
        }

        assert_eq!(
            found,
            Vec::<String>::new(),
            "only src/cnn/ops.rs may call candle"
        );
        assert!(files.contains(&ops) && files.contains(&src.join("cnn").join("mod.rs")));
        assert!(files.len() > 30, "only {} source files found", files.len());

        // The scan finds a use, and skips comments.
        assert_eq!(candle_mentions("use candle_core::Tensor;\n"), vec![1]);
        assert_eq!(candle_mentions("let t = candle::Tensor::zeros;\n"), vec![1]);
        assert!(candle_mentions("// candle_core is wrapped by ops.rs\n").is_empty());
    }

    #[test]
    fn ops_exposes_no_candle_type() {
        let code: Vec<&str> = production_source().lines().map(code_of).collect();
        let joined = code.join("\n");
        let mut found = Vec::new();

        // Public signatures: from `pub fn` to the opening brace.
        for (start, _) in joined.match_indices("pub fn ") {
            let signature = &joined[start..];
            let signature = &signature[..signature.find('{').unwrap_or(signature.len())];
            if signature.contains("candle") {
                found.push(signature.lines().next().unwrap_or("").to_string());
            }
        }

        // Public items and fields; tuple structs keep their field private.
        for line in &code {
            let line = line.trim_start();
            let public_item = line.starts_with("pub ") && !line.starts_with("pub fn ");
            let open_field = line.starts_with("pub struct ") && line.contains("(pub ");
            let other = !line.starts_with("pub struct ") && line.contains("candle");
            if public_item && (open_field || other || line.starts_with("pub use ")) {
                found.push(line.to_string());
            }
        }

        assert_eq!(found, Vec::<String>::new());
    }

    /// Largest absolute difference of two f64 tensors of the same shape.
    fn max_gap(ours: &Tensor, theirs: &candle::Tensor) -> f64 {
        assert_eq!(ours.dims(), theirs.dims());
        let theirs = Tensor(theirs.clone());

        f64_values(ours)
            .iter()
            .zip(f64_values(&theirs))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn replacements_match_candle_forward() {
        let mut c = Cases::new(7);
        let mut gaps: Vec<(String, f64)> = Vec::new();

        // W1: candle's conv1d forward is right for a contiguous kernel.
        for (padding, stride, dilation, groups, len) in [
            (1, 1, 1, 1, 9),
            (1, 2, 1, 1, 10),
            (2, 1, 2, 1, 9),
            (1, 1, 1, 2, 9),
            (1, 2, 2, 2, 11),
        ] {
            let x = tensor_of(&c.plain(&[2, 4, len]));
            let kernel = tensor_of(&c.plain(&[4, 4 / groups, 3]));
            let ours = conv1d(&x, &kernel, padding, stride, dilation, groups).unwrap();
            let theirs =
                x.0.conv1d(&kernel.0, padding, stride, dilation, groups)
                    .unwrap();
            gaps.push((
                format!("conv1d {padding} {stride} {dilation} {groups}"),
                max_gap(&ours, &theirs),
            ));
        }

        let x = tensor_of(&c.plain(&[2, 3, 9]));
        let kernel = tensor_of(&c.plain(&[4, 3, 3]));
        let ours = causal_conv1d(&x, &kernel, 2, 1).unwrap();
        let theirs =
            x.0.pad_with_zeros(2, 4, 0)
                .unwrap()
                .conv1d(&kernel.0, 0, 1, 2, 1)
                .unwrap();
        gaps.push(("causal_conv1d".into(), max_gap(&ours, &theirs)));

        // W2: the crop path against candle's conv2d.
        for (h, w, padding, stride, dilation) in
            [(7, 8, 1, 2, 1), (9, 10, 1, 2, 2), (5, 6, 0, 2, 1)]
        {
            let x = tensor_of(&c.plain(&[2, 3, h, w]));
            let side = if padding == 0 { 1 } else { 3 };
            let kernel = tensor_of(&c.plain(&[4, 3, side, side]));
            let ours = conv2d(&x, &kernel, padding, stride, dilation, 1).unwrap();
            let theirs = x.0.conv2d(&kernel.0, padding, stride, dilation, 1).unwrap();
            gaps.push((format!("conv2d {h}x{w}"), max_gap(&ours, &theirs)));
        }

        // W3 and W4 against candle's pooling, which takes no padding.
        for (h, w, kernel, stride) in [(7, 7, 2, 2), (7, 8, 3, 2), (5, 6, 3, 1), (7, 9, 2, 2)] {
            let x = tensor_of(&c.plain(&[2, 3, h, w]));
            let ours = max_pool2d(&x, kernel, stride, 0).unwrap();
            let theirs = x.0.max_pool2d_with_stride(kernel, stride).unwrap();
            gaps.push((
                format!("max_pool2d {h}x{w} {kernel} {stride}"),
                max_gap(&ours, &theirs),
            ));

            let ours = avg_pool2d(&x, kernel, stride, 0).unwrap();
            let theirs = x.0.avg_pool2d_with_stride(kernel, stride).unwrap();
            gaps.push((
                format!("avg_pool2d {h}x{w} {kernel} {stride}"),
                max_gap(&ours, &theirs),
            ));
        }

        // W5 against candle's upsampling.
        let x = tensor_of(&c.plain(&[2, 3, 3, 4]));
        let ours = upsample_nearest2d(&x, 2).unwrap();
        let theirs = x.0.upsample_nearest2d(6, 8).unwrap();
        gaps.push(("upsample_nearest2d".into(), max_gap(&ours, &theirs)));

        // W6 against candle's slow layer norm over the last axis; it rounds eps to f32.
        let x = tensor_of(&c.plain(&[2, 4, 3, 3]));
        let gamma = tensor_of(&c.uniform(&[4], 0.5, 1.5));
        let beta = tensor_of(&c.plain(&[4]));
        let eps = f64::from(1e-5f32);
        let ours = layer_norm_channels(&x, &gamma, &beta, eps).unwrap();
        let nhwc = x.0.permute((0, 2, 3, 1)).unwrap().contiguous().unwrap();
        let theirs = candle_nn::ops::layer_norm_slow(&nhwc, &gamma.0, &beta.0, 1e-5)
            .and_then(|out| out.permute((0, 3, 1, 2)))
            .unwrap();
        gaps.push(("layer_norm_channels".into(), max_gap(&ours, &theirs)));

        // W7 against candle's GELU; only its backward pass is off.
        let x = tensor_of(&c.uniform(&[2, 3, 4, 4], -3.0, 3.0));
        gaps.push((
            "gelu_tanh".into(),
            max_gap(&gelu_tanh(&x).unwrap(), &x.0.gelu().unwrap()),
        ));
        gaps.push((
            "gelu_erf".into(),
            max_gap(&gelu_erf(&x).unwrap(), &x.0.gelu_erf().unwrap()),
        ));

        // The 1-D pools against candle's 2-D pools with a 1 x k window on a height of 1.
        for (len, kernel, stride) in [(9, 2, 2), (10, 3, 2), (11, 4, 4)] {
            let x = tensor_of(&c.plain(&[2, 3, len]));
            let rows = x.0.unsqueeze(2).unwrap();
            let theirs = rows
                .max_pool2d_with_stride((1, kernel), (1, stride))
                .unwrap()
                .squeeze(2)
                .unwrap();
            gaps.push((
                format!("max_pool1d {len} {kernel} {stride}"),
                max_gap(&max_pool1d(&x, kernel, stride).unwrap(), &theirs),
            ));
            let theirs = rows
                .avg_pool2d_with_stride((1, kernel), (1, stride))
                .unwrap()
                .squeeze(2)
                .unwrap();
            gaps.push((
                format!("avg_pool1d {len} {kernel} {stride}"),
                max_gap(&avg_pool1d(&x, kernel, stride).unwrap(), &theirs),
            ));
        }

        // The global average pool against candle's mean over height and width.
        let ours = global_avg_pool(&x).unwrap();
        gaps.push((
            "global_avg_pool".into(),
            max_gap(&ours, &x.0.mean((2, 3)).unwrap()),
        ));

        for (name, gap) in &gaps {
            eprintln!("forward gap to candle: {name}: {gap:.1e}");
            assert!(
                *gap <= FORWARD_LIMIT,
                "{name} differs from candle by {gap:e}"
            );
        }
    }

    #[test]
    fn max_pool_ties_split_the_gradient() {
        // Four 2x2 windows with 1, 2, 3 and 4 tied maxima.
        #[rustfmt::skip]
        let image = vec![
            1.0, 0.0, 2.0, 2.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 0.0, 4.0, 4.0,
        ];
        let param = Param::new(&f64_tensor(image, &[1, 1, 4, 4])).unwrap();
        let out = max_pool2d(&param.tensor(), 2, 2, 0).unwrap();
        let loss = Tensor(out.0.sum_all().unwrap());
        let grad = f64_values(&backward(&loss).unwrap().get(&param).unwrap());

        for (top, left) in [(0, 0), (0, 2), (2, 0), (2, 2)] {
            let window: f64 = [(0, 0), (0, 1), (1, 0), (1, 1)]
                .iter()
                .map(|(dy, dx)| grad[(top + dy) * 4 + left + dx])
                .sum();
            assert_eq!(window, 1.0, "window at ({top}, {left})");
        }
    }

    #[test]
    fn causal_conv1d_reads_no_later_input() {
        let mut c = Cases::new(8);
        let x = c.plain(&[1, 2, 12]);
        let kernel = tensor_of(&c.plain(&[3, 2, 3]));
        let before = causal_conv1d(&tensor_of(&x), &kernel, 2, 1).unwrap();

        // Change step 9 of both input channels.
        let mut changed = x.values.clone();
        changed[9] += 1.0;
        changed[12 + 9] += 1.0;
        let after = causal_conv1d(&f64_tensor(changed, &[1, 2, 12]), &kernel, 2, 1).unwrap();

        assert_eq!(before.dims(), [1, 3, 12]);
        let (before, after) = (f64_values(&before), f64_values(&after));
        for channel in 0..3 {
            let row = channel * 12;
            assert_eq!(
                before[row..row + 9],
                after[row..row + 9],
                "channel {channel}"
            );
            assert_ne!(before[row + 9], after[row + 9], "channel {channel}");
        }
    }

    #[test]
    fn batch_norm_tracks_running_statistics_and_evaluates_with_them() {
        let gamma = Tensor::from_f32(vec![2.0, 0.5], &[2]).unwrap();
        let beta = Tensor::from_f32(vec![0.1, -0.2], &[2]).unwrap();
        let mean = Tensor::from_f32(vec![0.0, 0.0], &[2]).unwrap();
        let var = Tensor::from_f32(vec![1.0, 1.0], &[2]).unwrap();
        let norm = BatchNorm::new(&gamma, &beta, &mean, &var, 1e-5).unwrap();

        // Shape (3, 2, 2): channel 0 holds 1..6 and channel 1 holds 10, 20, ..., 60.
        let values = vec![
            1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 50.0, 60.0,
        ];
        let x = Tensor::from_f32(values.clone(), &[3, 2, 2]).unwrap();
        norm.forward_train(&x).unwrap();

        // The batch has mean 3.5 and 35, and unbiased variance 3.5 and 350.
        let expected_mean = [0.1 * 3.5, 0.1 * 35.0];
        let expected_var = [0.9 + 0.1 * 3.5, 0.9 + 0.1 * 350.0];
        let running_mean = norm.running_mean().to_vec_f32().unwrap();
        let running_var = norm.running_var().to_vec_f32().unwrap();
        for c in 0..2 {
            assert!(
                (running_mean[c] - expected_mean[c]).abs() < 1e-5,
                "{running_mean:?}"
            );
            assert!(
                (running_var[c] - expected_var[c]).abs() < 1e-4,
                "{running_var:?}"
            );
        }

        let out = norm.forward_eval(&x).unwrap().to_vec_f32().unwrap();
        let (gamma, beta) = ([2.0f32, 0.5], [0.1f32, -0.2]);
        for (i, (&v, &o)) in values.iter().zip(&out).enumerate() {
            let c = (i / 2) % 2;
            let expected =
                (v - running_mean[c]) / (running_var[c] + 1e-5).sqrt() * gamma[c] + beta[c];
            assert!(
                (o - expected).abs() < 1e-5,
                "entry {i}: {o} against {expected}"
            );
        }
    }

    #[test]
    fn normal_values_come_from_the_seeded_stream() {
        let mut a = ChaCha8Rng::seed_from_u64(5);
        let mut b = ChaCha8Rng::seed_from_u64(5);
        let first = Tensor::normal(&mut a, &[2, 3], 0.5).unwrap();
        let again = Tensor::normal(&mut b, &[2, 3], 0.5).unwrap();
        assert_eq!(first.to_vec_f32().unwrap(), again.to_vec_f32().unwrap());
        assert_eq!(first.dims(), [2, 3]);

        // The values are 0.5 times draws of util::normal_f32, in row-major order.
        let mut c = ChaCha8Rng::seed_from_u64(5);
        let expected: Vec<f32> = (0..6).map(|_| 0.5 * normal_f32(&mut c)).collect();
        assert_eq!(first.to_vec_f32().unwrap(), expected);

        // The stream moves on, so the next tensor differs.
        let next = Tensor::normal(&mut a, &[2, 3], 0.5).unwrap();
        assert_ne!(next.to_vec_f32().unwrap(), expected);
    }

    #[test]
    fn params_take_copies_and_gradients_skip_unused_params() {
        let init = Tensor::from_f32(vec![1.0, 2.0], &[2]).unwrap();
        let used = Param::new(&init).unwrap();
        let unused = Param::new(&init).unwrap();

        let loss = Tensor(used.tensor().0.sqr().unwrap().sum_all().unwrap());
        let grads = backward(&loss).unwrap();
        assert_eq!(
            grads.get(&used).unwrap().to_vec_f32().unwrap(),
            vec![2.0, 4.0]
        );
        assert!(grads.get(&unused).is_none());
        assert!(
            backward(&used.tensor()).is_err(),
            "the loss must be a scalar"
        );

        used.set(&Tensor::from_f32(vec![5.0, 6.0], &[2]).unwrap())
            .unwrap();
        assert_eq!(used.tensor().to_vec_f32().unwrap(), vec![5.0, 6.0]);
        assert_eq!(init.to_vec_f32().unwrap(), vec![1.0, 2.0]);
        assert_eq!(unused.tensor().to_vec_f32().unwrap(), vec![1.0, 2.0]);
        assert!(
            used.set(&Tensor::from_f32(vec![1.0], &[1]).unwrap())
                .is_err()
        );
    }

    #[test]
    fn bad_shapes_and_settings_are_rejected() {
        let zeros = |shape: &[usize]| {
            let count = shape.iter().product();
            Tensor::from_f32(vec![0.0; count], shape).unwrap()
        };
        let x = zeros(&[2, 4, 5, 5]);

        assert!(Tensor::from_f32(vec![0.0; 5], &[2, 3]).is_err());
        assert!(
            out_len(5, 0, 7, 1, 1).is_err(),
            "a window wider than the input"
        );
        assert!(out_len(5, 0, 0, 1, 1).is_err(), "an empty kernel");
        assert_eq!(out_len(7, 1, 3, 2, 1).unwrap(), 4);
        assert_eq!(out_len(8, 1, 3, 2, 2).unwrap(), 3);

        assert!(
            conv2d(&x, &zeros(&[6, 2, 3, 3]), 1, 1, 1, 3).is_err(),
            "3 groups, 4 channels"
        );
        assert!(
            conv2d(&x, &zeros(&[4, 3, 3, 3]), 1, 1, 1, 1).is_err(),
            "kernel reads 3 channels"
        );
        assert!(
            conv2d(&x, &zeros(&[4, 4, 3, 3]), 1, 0, 1, 1).is_err(),
            "stride 0"
        );
        assert!(
            conv2d(&x, &zeros(&[4, 4, 7, 7]), 0, 1, 1, 1).is_err(),
            "kernel wider than input"
        );
        assert!(
            conv1d(&x, &zeros(&[4, 4, 3]), 1, 1, 1, 1).is_err(),
            "a 4-D input"
        );
        assert!(conv_transpose2d(&x, &zeros(&[4, 2, 2, 2]), 0, 2, 2, 1).is_err());
        assert!(
            max_pool2d(&x, 2, 2, 2).is_err(),
            "padding above half the kernel"
        );
        assert!(
            avg_pool2d(&x, 3, 1, 2).is_err(),
            "padding above half the kernel"
        );
        assert!(max_pool1d(&x, 2, 2).is_err(), "a 4-D input");
        assert!(
            max_pool1d(&zeros(&[2, 3, 4]), 5, 5).is_err(),
            "a window wider than the input"
        );
        assert!(avg_pool1d(&zeros(&[2, 3, 4]), 0, 1).is_err(), "an empty window");
        assert!(
            broadcast_add(&x, &zeros(&[1, 3, 1, 1])).is_err(),
            "3 biases for 4 channels"
        );
        assert!(mse(&zeros(&[2, 3]), &zeros(&[3, 2])).is_err());
        assert!(mse(&zeros(&[0, 3]), &zeros(&[0, 3])).is_err(), "no values");
        assert!(upsample_nearest2d(&x, 0).is_err());
        assert!(channel_shuffle(&x, 3).is_err());
        assert!(group_norm(&x, &zeros(&[4]), &zeros(&[4]), 3, 1e-5).is_err());
        assert!(layer_norm_channels(&x, &zeros(&[3]), &zeros(&[4]), 1e-5).is_err());
        assert!(linear(&zeros(&[2, 5]), &zeros(&[3, 4]), &zeros(&[3])).is_err());
        assert!(
            cross_entropy(&zeros(&[2, 3]), &[0, 3]).is_err(),
            "label 3 of 3 classes"
        );
        assert!(
            cross_entropy(&zeros(&[2, 3]), &[0]).is_err(),
            "one label for two rows"
        );
        assert!(global_avg_pool(&zeros(&[2, 3])).is_err());
        assert!(reshape(&x, &[2, 3]).is_err());

        // A candle error becomes one line, without candle's backtrace.
        let message = cat(&[&zeros(&[2, 3]), &zeros(&[3, 3])], 1)
            .expect_err("rows of 2 and 3 do not join")
            .to_string();
        assert!(message.starts_with("invalid: tensor: "), "{message}");
        assert!(!message.contains('\n'), "{message}");

        let one = zeros(&[1, 2, 1, 1]);
        let norm = BatchNorm::new(&zeros(&[2]), &zeros(&[2]), &zeros(&[2]), &zeros(&[2]), 1e-5);
        assert!(
            norm.unwrap().forward_train(&one).is_err(),
            "one value per channel"
        );
    }
}
