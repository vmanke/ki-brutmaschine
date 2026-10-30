# 04 CNN families, part B

Status: ready-for-agent
Blocked by: 02
Spec: `../spec.md`

## What to do

Blocks and one scaled-down base model each, within the limits:

- DenseNet (dense block, transition layer);
- Inception (parallel 1×1, 3×3, 5×5 and pooling branches, concatenated);
- SqueezeNet (Fire module);
- ShuffleNet (group convolution, channel shuffle);
- ConvNeXt (depthwise 7×7, LayerNorm, inverted MLP, GELU);
- TCN (causal dilated 1-D convolution with residuals), for the 1-D task;
- U-Net (encoder, decoder, concatenated skips; transposed convolution or upsampling),
  for grid-to-grid outputs.

## Done when

- The same tests as ticket 03 for each block and base model.
- The TCN learns the 1-D motif task above chance; U-Net maps a generated grid to its
  mirror image above chance (a stand-in until ticket 06).

Cost: about 15 to 30 dollars.

## Comments
