# 03 CNN families, part A

Status: ready-for-agent
Blocked by: 02
Spec: `../spec.md`

## What to do

Blocks and one scaled-down base model each, within the limits:

- LeNet-5 (5×5 convolution, pooling, dense head);
- AlexNet style (large strided first convolution);
- VGG (stacks of 3×3);
- ResNet (basic and bottleneck residual blocks);
- Xception and MobileNet (depthwise separable convolution);
- MobileNetV2 (inverted residual, linear bottleneck, ReLU6);
- SE (squeeze and excitation as an option on any block);
- dilated convolution.

## Done when

- Each block has a test of its output shape and parameter count against a hand count,
  and a test that its gradients are finite and reduce the loss on a toy batch.
- Each base model passes the shape check and learns the shapes task above chance.

Cost: about 15 to 30 dollars.

## Comments
