# CNN base models in the MLP mode

Status: ready-for-agent (approved by the owner on 2026-10-30: all families, generated
images first and ARC grids second, an ML library; budget about 80 to 160 dollars). Work
starts after the counted Runs of step 1 have finished, so that no build slows them.

## Goal

The MLP mode breeds small dense networks today (`src/genome.rs`, `src/nn.rs`). Its genes
are the hidden widths, one of 4 activations, one of 3 optimizers and a few training
values. This feature adds convolutional networks (CNNs) as a second kind of model. The
mutations can then switch between the known CNN families, change their parts, and find
mixed forms.

"All possible kinds" cannot be built literally; there are infinitely many. So the genome
is a kit of building blocks, and the known families are special cases of it. Each family
also enters as a scaled-down base model (seed) of the population.

## The families and their sources

Peer-reviewed unless marked as a preprint.

| Family | Building block | Source |
|---|---|---|
| LeNet-5 | 5×5 convolution, pooling, dense head | LeCun, Bottou, Bengio and Haffner 1998, Proc. IEEE 86(11) |
| AlexNet | large strided first convolution, ReLU | Krizhevsky, Sutskever and Hinton 2012, NeurIPS |
| VGG | stacks of 3×3 convolutions | Simonyan and Zisserman 2015, ICLR |
| GoogLeNet | Inception block: parallel 1×1, 3×3, 5×5 and pooling branches | Szegedy et al. 2015, CVPR |
| ResNet | residual block, basic and bottleneck | He, Zhang, Ren and Sun 2016, CVPR |
| DenseNet | dense block (concatenation), transition | Huang, Liu, van der Maaten and Weinberger 2017, CVPR |
| Xception | depthwise separable convolution | Chollet 2017, CVPR |
| MobileNet | depthwise separable convolution | Howard et al. 2017, arXiv:1704.04861 (preprint) |
| MobileNetV2 | inverted residual, linear bottleneck | Sandler et al. 2018, CVPR |
| ShuffleNet | group convolution, channel shuffle | Zhang, Zhou, Lin and Sun 2018, CVPR |
| SqueezeNet | Fire module (squeeze 1×1, expand 1×1 and 3×3) | Iandola et al. 2016, arXiv:1602.07360 (preprint) |
| SENet | squeeze-and-excitation on any block | Hu, Shen and Sun 2018, CVPR |
| ConvNeXt | depthwise 7×7, LayerNorm, inverted MLP, GELU | Liu et al. 2022, CVPR |
| U-Net | encoder and decoder with concatenated skips | Ronneberger, Fischer and Brox 2015, MICCAI |
| Dilated CNN | dilated convolution | Yu and Koltun 2016, ICLR |
| TCN | causal dilated 1-D convolution with residuals | Bai, Kolter and Koltun 2018, arXiv:1803.01271 (preprint) |

Normalization genes: none, batch norm (Ioffe and Szegedy 2015, ICML), layer norm (Ba,
Kiros and Hinton 2016, arXiv:1607.06450, preprint), group norm (Wu and He 2018, ECCV).
Evolving CNN architectures has precedent: Real et al. 2017 (ICML) and 2019 (AAAI); survey:
Elsken, Metzen and Hutter 2019, JMLR 20(55).

## Design

- **Model genome.** A model is either the existing MLP genome or a CNN genome. Existing
  archives must stay readable (serde default for the new field).
- **CNN genome.** An input stem, a list of stages, and a head. A stage has a block type
  (one of the families above), its width, kernel size, stride, dilation, groups, an
  optional SE part, a normalization and an activation. Downsampling is a strided
  convolution, max pooling or average pooling. The head is global average pooling plus a
  dense layer, flatten plus dense layers (LeNet, VGG), or a 1×1 convolution per cell
  (U-Net, for grid-to-grid tasks). Activations: the existing four plus SiLU and ReLU6.
- **Shape check.** Every genome is checked before training: shapes fit, and the
  parameter count and the training FLOPs stay within `Limits` (`max_params`,
  `max_train_flops`). MLPs and CNNs share these limits, so neither wins by size. A CNN's
  estimated training time must also fit the time that `max_train_flops` buys at 1 GFLOP/s,
  so that its training ends by the FLOP limit, not by the clock (see Reproducibility).
- **Mutations.** Switch a stage's family where the shapes allow it; insert or remove a
  stage; change width, kernel, stride, dilation or groups; toggle SE, residual and
  normalization; change downsampling or activation. Crossover exchanges whole stages at
  matching resolutions. Every child passes the shape check, or it is redrawn.
- **Base models.** One scaled-down seed per family, within the limits, enters the first
  generation next to random genomes.
- **Promotion of winners.** Winners can be fed back upward (owner, 2026-10-30): the two
  best models of a run by their hidden score, and only if they beat the run's best base
  model, are promoted into `candidates/models/` through a draft pull request that a
  human merges, as for Cascades (ADR 0004, ADR 0005). Later runs start with the merged
  models next to the base models of the families, so the pool of base models grows.
- **Library.** candle (`candle-core`, `candle-nn`, Hugging Face), from GitHub at commit
  `9515d410e3af803fd45efebfef8137f47d85a63d` of its main branch (version 0.11.0), with
  default features only (CPU), pinned in `Cargo.lock` (ADR 0007). The release 0.11.0 on
  crates.io is rejected: it requires zip 8.6.0, which has three open advisories; the commit
  uses zip 9.1.3. candle runs on the CPU in Rust and has every operation the families need:
  1-D and 2-D convolution with stride, dilation and groups, transposed convolution, max and
  average pooling, upsampling, batch, layer and group norm, and automatic gradients. Its CPU
  backward pass is wrong in 17 of 60 checked cases, through 7 defects (D1 to D7); an eighth
  defect, D8, is that candle cannot seed its CPU generator. `src/cnn/ops.rs` is the only file
  that calls candle; it replaces each defective operation with one built from correct ones
  (W1 to W8) and checks every operation it offers against finite differences. Ticket 04
  found a ninth defect, D9: `conv_transpose1d` has no backward pass, and its forward pass is
  wrong when the batch and the input channels both exceed 1. W9 runs `conv_transpose2d` on a
  height of 1 instead (ADR 0007, addendum of 2026-10-31). burn 0.22.1
  (tracel-ai) has a larger layer zoo, but it needs Rust 1.95, brings 133 new crates against
  candle's 103, and its lockfile fails `cargo audit` as well. burn was rejected for these
  three reasons alone: it was not built, so its gradients, determinism and speed were not
  checked. Ticket 01 settled this (Comments there).
- **Reproducibility.** The project compares logs byte for byte. CNN training gives the
  same bits at any thread count on one machine: ticket 01 checked 19 thread settings. The
  tests of ticket 02 compare every step's loss and the model file across runs and rayon
  pools of 1 and 4 threads, and across child processes with `RAYON_NUM_THREADS` at 1 and
  3, the variable from which candle takes the threads of its matrix products. So no fixed
  thread count is needed, and candidates train in parallel as MLPs do. The bits are equal
  only while no candidate hits a wall-clock limit. So the shape check also estimates a
  CNN's training time (`cnn::shape`, "Time") and holds it to `max_train_flops` at
  1 GFLOP/s, 20 s at the default limits, below `max_candidate_seconds` (30 s); a busier or
  slower machine can still reach the clock. Other CPUs may give other bits, because gemm
  picks its kernel (with or without fused multiply-add) and its cache blocking at run time
  (ADR 0007). Tests therefore compare results within one machine only. The scores of a CNN
  also depend on the chunk sizes of `score::evaluate`, which are part of this contract.
- **Security.** Before candle entered `Cargo.toml`, `cargo audit` (RustSec, cargo-audit
  0.22.2, advisory database of 2026-10-30) checked it and every new transitive dependency;
  a crate with an open advisory is not admitted. With the pinned commit the audit of the
  repository exits with 0 over 164 packages. Its one warning is RUSTSEC-2024-0436: paste
  1.0.15, a compile-time macro crate of gemm, is unmaintained. No crate needs a C or C++
  compiler; `cargo tree -i cc` finds no `cc`. The lockfile pins the versions. `cargo audit`
  fetches its database from GitHub, so the RustSec website, which is outside this
  session's network allow-list, is not needed.

## Tasks

- **First, generated images and sequences** (no download, seeded like the existing
  tasks, with train, validation and hidden splits): shapes in 16×16 images (square,
  circle, triangle, cross, at random position and size, with noise); bars and stripes
  (orientation); counting blobs (1 to 4); mirror symmetry; and for the 1-D families a
  motif anywhere in a sequence.
- **Second, ARC grids:** grid-to-grid tasks, one colour class per cell, up to 30×30, for
  U-Net and fully convolutional models. The sources, commits and licences are pinned in
  the project already (ARC-AGI-1 and ARC-AGI-2 under Apache-2.0, re-arc and ConceptARC
  under MIT), and `src/arc_data.rs` loads them with a SHA-256 manifest. Training uses the
  ARC-AGI-1 training tasks with their re-arc generations; both evaluation sets stay held
  out. The converter `tools/rearc_to_jsonl.py` is missing and is rewritten in ticket 06.

## Tickets and pull requests

1. `issues/01-library-spike.md`: candle, advisory check, determinism, speed.
2. `issues/02-model-seam-and-grid-tasks.md`: the model genome, shape check, limits, and
   the generated image and sequence tasks.
3. `issues/03-families-a.md`: LeNet, AlexNet, VGG, ResNet, Xception and MobileNet,
   MobileNetV2, SE, dilated convolution.
4. `issues/04-families-b.md`: DenseNet, Inception, SqueezeNet, ShuffleNet, ConvNeXt,
   TCN, U-Net.
5. `issues/05-mutations-and-integration.md`: mutation, crossover, base models in the
   population, CLI, README (German and English), ADR 0008, CONTEXT.md.
6. `issues/06-arc-grids.md`: ARC grid tasks.
7. `issues/07-promotion-of-winners.md`: winners become base models of later runs.

Pull requests: tickets 01 and 02; then 03 to 05 and 07; then 06. Each is reviewed once by an
independent reviewer; further rounds only on the owner's request.

## Cost estimate

About 90 to 180 dollars of model usage in total: 01 about 8 to 15, 02 about 10 to 20, 03
and 04 about 15 to 30 each, 05 about 10 to 20, 06 about 10 to 20, 07 about 10 to 20, docs and one review per
pull request included. Every step that would exceed its range is announced first.

## Comments
