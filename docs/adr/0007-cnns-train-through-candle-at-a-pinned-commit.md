# CNNs train through candle at a pinned commit, and only one file calls it

The CNN feature (`.scratch/cnn-basismodelle/spec.md`) adds convolutional networks as a
second kind of model in the MLP mode. They need convolutions, pooling, norms and automatic
gradients; the hand-written MLP of `src/nn.rs` has none of these. Ticket 01 tested candle
(Hugging Face) for this in a throwaway crate. A second, independent check repeated the
tests, and a third run repeated them at the commit chosen here. This record decides the
library, its version, how the crate reaches it, and what reproducibility we can promise.
The evidence is in ticket 01 (`.scratch/cnn-basismodelle/issues/01-library-spike.md`,
Comments).

**candle from a pinned commit, not from the release.** `Cargo.toml` takes candle-core and
candle-nn from GitHub at commit 9515d410e3af803fd45efebfef8137f47d85a63d of the main branch,
which still calls itself 0.11.0. We use default features only: CPU, no MKL, no CUDA, no
Metal. `Cargo.lock` pins the commit, and zip at 9.1.3: the resolver would take zip 9.2.0,
but 9.1.3 is the version the run at the commit built and audited. The release 0.11.0 on
crates.io is rejected: it requires zip 8.6.0, not as an option, and zip 8.6.0 has three
open advisories. RUSTSEC-2026-0367 (CVSS 7.3) keeps SUID and SGID bits on extraction,
RUSTSEC-2026-0368 creates symlinks that point outside the target directory, and
RUSTSEC-2026-0369 (CVSS 9.1) applies Unix permissions through symlinks. All three are fixed
from zip 9.0.0 on, and the spec admits no crate with an open advisory. candle uses zip only
in its readers and writers of NumPy and PyTorch files (`npy.rs`, `pickle.rs`), which we
never call; that is a reading of the source, and it does not change the rule.

**The supply chain passes the audit.** cargo-audit 0.22.2 with the advisory database at
commit 2295466c (2026-10-30, 1395 advisories) scans the 164 packages of the lockfile and
exits with 0. Its one warning is RUSTSEC-2024-0436: paste 1.0.15 is unmaintained. paste is a
macro crate that gemm, candle's matrix library, uses at compile time; an unmaintained crate
is a warning, not a vulnerability. The lockfile grows from 40 to 164 packages, and the 40
old ones keep their versions. The compiled tree for x86-64 Linux grows from 34 to 137
crates, so 103 are new. All 103 have permissive licences: MIT or Apache-2.0 (59 in several
spellings), MIT alone (25), Apache-2.0 alone (4), Unicode-3.0 (4), Zlib or Apache-2.0 or
MIT (4), Unlicense or MIT (2), Apache-2.0 with LLVM exception or Apache-2.0 or MIT (2),
Zlib (1), Apache-2.0 or BSL-1.0 (1), and BSD-2-Clause or Apache-2.0 or MIT (1). There is no
`cc` crate, so the build needs no C or C++ compiler; the release needed a C compiler for the
Oniguruma code of tokenizers. No new crate is a network library: none of ureq, reqwest,
hyper, hf-hub, rustls, native-tls, openssl or tokio is in the tree, so ADR 0001 holds. No
new crate declares a Rust version above 1.88 (zip). The repository declares 1.89; we build
with 1.97.0 only. The two crates amend invariant I1 of
the Stage 1 contract (`docs/STAGE1_CONTRACT.md`, no new dependencies) and invariant I1 of
the backend contract (`docs/STAGE2_BACKEND_CONTRACT.md`: the core crate gains no new
dependency of any kind). Errata E15 (`docs/STAGE1_CONTRACT_ERRATA.md`) records this, and
both contracts point to it. The test of I1,
`t_sec_3_dependencies_unchanged` in `tests/learner_security.rs`, now admits candle-core and
candle-nn, and only from this commit, without a branch, tag, path or feature list.

**burn is the alternative, and it does not fit better.** burn 0.22.1 with the features std,
ndarray, autodiff and optim has the larger layer zoo. But it declares Rust 1.95, above our
1.89. Its compiled tree has 144 crates, 133 of them new to the Brutmaschine, against 103 for
candle. Its lockfile of 608 packages fails cargo audit as well: 4 vulnerabilities (zip 0.6.6
with the same three advisories, zstd-safe 5.0.2 with RUSTSEC-2026-0403) and 9 warnings. Those
crates come in only through tch and torch-sys, which these features do not compile, but the
lockfile would carry them. burn was compared on these points only: it was not built, so
its gradients, determinism and speed were not checked.

**candle has eight defects, and each has a replacement.** The spike checked every operation
the families need, in f64, against a four-point central difference with step 1e-3. An entry
fails when |a - n| / max(|a|, |n|, 1e-3 times the largest numeric gradient) reaches 1e-6.
Of 60 native cases, 43 are correct (the worst at 2.0e-9) and 17 are wrong or fail. They
trace back to seven defects, D1 to D7. An eighth defect, D8, is that candle cannot seed its
CPU generator; it is not one of the 60 cases. The run at the pinned commit found the same 17, and all 30
cases of the replacements correct (the worst at 5.9e-10). The lines below are paths in the
candle repository at that commit.

| Defect | Where | What goes wrong | Replacement |
|---|---|---|---|
| D1 | `candle-core/src/cpu_backend/mod.rs:2861-2869`, `candle-core/src/backprop.rs:264` | `conv1d` copies a strided kernel but multiplies with the original; the backward pass passes such a kernel, so the kernel gradient is wrong when batch and output channels per group exceed 1 | W1: `conv1d` and `causal_conv1d` as a `conv2d` of height 1, with explicit padding |
| D2 | `candle-core/src/backprop.rs:289` | the backward pass of `conv2d` takes the output padding from the height alone; with a stride, unequal remainders of height and width fail with a shape mismatch | W2: pad explicitly and crop the rows and columns no window reads |
| D3 | `backprop.rs:363`, `:373-374` | `max_pool2d` has no backward pass unless kernel equals stride; where it has one, a single maximum of a 2x2 window gets a quarter of its gradient | W3: running maximum over the shifted views; ties split the gradient |
| D4 | `backprop.rs:348`, `:352` | `avg_pool2d` has no backward pass unless kernel equals stride; where it has one, the gradient lands on the wrong cells when the size is not a multiple of the kernel | W4: crop, then candle, when kernel equals stride; otherwise a sum of shifted views |
| D5 | `backprop.rs:388`, `:409` | the backward pass of `upsample_nearest1d` and `2d` assigns the gradient instead of adding it, so an input that also feeds another branch can lose that gradient | W5: upsampling by reshape and broadcast |
| D6 | `candle-nn/src/layer_norm.rs:118-120`, `candle-nn/src/ops.rs:944` | `LayerNorm` runs a fused kernel without a backward pass; in f32 the gradient stops without an error, and f64 is refused (`ops.rs:772`) | W6: layer norm over the channels from mean and variance |
| D7 | `backprop.rs:613-615`, `:628-630` | the backward passes of `gelu` and `gelu_erf` use constants rounded to 6 digits; the gradient is off by up to 2.5e-7 and 1.7e-7 | W7: GELU built from `tanh` and from `erf` |
| D8 | `candle-core/src/cpu_backend/mod.rs:3185` | `set_seed` fails on the CPU, so candle's random numbers cannot be reproduced | W8: every random value from our own ChaCha8 stream |

D2 hits non-square grids such as ARC's, D1 every 1-D model such as TCN, and D5 only an
upsampled tensor that also feeds another branch; a plain U-Net decoder is not hit. D7 is
of the order of f32 rounding; W7 removes it, so that every operation passes the same check.

**Only `src/cnn/ops.rs` calls candle.** It wraps candle's tensor in its own `Tensor` type,
whose field is private, so no other code can reach a candle operation. It offers the eight
replacements and thin wrappers of the native operations found correct: `conv2d` at stride
1 or with equal remainders, grouped and depthwise, `conv_transpose2d`, batch norm in
training mode, group norm, concatenation, broadcast multiply, channel shuffle, residual add,
reshape and flatten, a dense layer, softmax cross-entropy, ReLU, ReLU6, SiLU, tanh, sigmoid,
sine and global average pooling. Its tests run the f64 gradient check on every public
operation, at the shapes that broke candle: odd and non-square sizes, stride 2, dilation 2,
groups, kernels unequal to the stride, and inputs that feed two branches. They compare each
replacement's forward pass with candle's own. They run candle's own operation at each of
the defects D1 to D7 and require it to fail the same check, and they require `set_seed` to
fail (D8); so a candle commit that fixes a defect shows up there. Two scans of the sources
fail if any other file under `src/` names candle, or if `ops.rs` makes a call with a known
defect outside the replacement that needs it. A candle error becomes `BrutError::Invalid`
without candle's backtrace, because a backtrace would make a logged message differ between
machines.

**Training is byte-equal at any thread count, on one machine.** The spike trained a small
CNN in f32 for 50 AdamW steps: 2736 parameters, with a convolution, batch norm, a depthwise
and a pointwise convolution with a residual add, squeeze-and-excitation, a concatenation,
max and average pooling and a dense head. It ran under 19 thread settings: RAYON_NUM_THREADS
from 1 to 4 and unset, installed rayon pools of other sizes, candle's `Device::with_context`,
CANDLE_NUM_THREADS, and gemm's threading threshold at 1, so that every product splits. Each
variant gave one hash of the losses and one of the final state, at the release and again
at the pinned commit. At the release, the check repeated this with a second network of
547 000 parameters under 18 settings, and with copies trained in parallel; all matched the
1-thread run. From the source: gemm never splits the summed dimension, and the parallel
loops write disjoint outputs. So no thread setting is needed, and candidates may train in
parallel. The bits at the pinned commit differ from those of the release: the final loss of
the same run is 0.238868 against 0.238249. Results recorded with one candle version
therefore cannot be compared byte for byte with another; a change of the commit is a change
of the numbers. Other CPUs may give other bits. gemm chooses its kernel at run time, FMA
or scalar (gemm-common `gemm.rs:1018`). For products with more than 64 rows or columns it
also sizes its blocks from the L1 cache it detects (gemm-common `cache.rs:514-570`; read,
not tested). Tests therefore compare results within one machine only. The CNN training
of the crate keeps this property. Its tests compare the loss of every step and the model
file of two runs in a row and in rayon pools of 1 and 4 threads. They compare the same
fingerprints in child processes with `RAYON_NUM_THREADS` at 1 and 3: candle reads that
variable when a matrix product starts, not the rayon pool around it, so only a child
process varies the threads of gemm. And they compare the candidates and model files of a
run with 1 and 2 worker threads, and of `brut run --no-hidden` in child processes with
`RAYON_NUM_THREADS` at 1 and 3 (ticket 02).

The equal bits need one more condition: no candidate may hit a wall-clock limit. When
`max_candidate_seconds` stops a training, the epochs done depend on the load of the
machine. For an MLP the FLOP limit always comes first (13 GFLOP/s by the count). For a CNN
it does not by itself: the time per counted FLOP varies about 70-fold between layouts
(0.08 GFLOP/s for depthwise layers of 256 groups, 0.15 for narrow layers at full size, 6
for wide ones, at one thread),
since candle spends most of the time of narrow, grouped and normed layers on copies and
calls, not on products. So the shape check also estimates a CNN's training time from its
plan (`cnn::shape`, "Time") and holds it to `max_train_flops` at 1 GFLOP/s, 20 s at the
default limits. The model comes from timing 77 CNNs on this machine; every measured time
stayed below 0.89 of the estimate. The slowest valid CNNs of seven slow layouts took 7 to
13 s at one thread. A busier or slower machine can still reach the 30 s, and then
`stopped_by_budget` marks the candidate.

**A training step is fast enough for the budget.** On this machine, with the experiment of
step 1 running and at nice 10, one step of that CNN on a batch of 64 images of 16x16
(forward, backward and the AdamW update) took 42 to 51 ms at 1 or 2 threads. A second
thread changed the time by a factor of 0.91 to 1.2. By our count a step costs about 77
MFLOP, so the default `max_train_flops` of 2e10 allows about 260 steps, about 12 s, below
the default `max_candidate_seconds` of 30 s. These times are from the release; the run at
the pinned commit did not time steps. This holds for that CNN only. Ticket 02 measured
CNNs at 0.08 to 6 GFLOP/s by the same count, so the time model above, not this estimate,
keeps every valid CNN within the budget.

Rejected: candle 0.11.0 from crates.io (zip 8.6.0 with three open advisories); the release
with a local patch for zip 9 (6 lines, compiled but never run, and a fork to maintain);
burn 0.22.1 (Rust 1.95, 133 new crates, and its lockfile fails cargo audit); a fixed number
of threads per candidate (not needed, since every setting tested gave the same bits); the
spike's other replacements, max pooling by window shares, depthwise convolution from
shifted views and grouped convolution as one block-diagonal kernel (candle's own `groups`
is correct, and at 1 thread fastest); candle_nn's `layer_norm_slow` (correct, but 28.1 ms
against 22.8 ms for W6).

Status: accepted on 2026-10-31. The owner approved an ML library for the CNN feature on
2026-10-30 (spec). Claude, the AI assistant, chose candle at this commit after the spike,
the independent check and the run at the commit.
