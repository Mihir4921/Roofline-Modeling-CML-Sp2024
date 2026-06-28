# GPU Roofline Modeling — ImageNet Training Analysis

A performance study that uses the **Roofline model** to characterize how convolutional neural networks behave when trained on different cloud GPUs, and to determine whether they are limited by raw compute or by memory bandwidth.

## Why this exists

Training deep-learning models in the cloud means choosing (and paying for) specific GPUs, but raw spec-sheet TFLOPS rarely tell you what you'll actually get. A model may leave a fast GPU idle because it's bottlenecked on memory traffic rather than compute. The Roofline model makes that trade-off visible: by plotting a workload's **arithmetic intensity** (FLOPs per byte of memory traffic) against the hardware's peak compute and peak bandwidth, you can see at a glance whether a given model on a given GPU is **compute-bound** or **memory-bound**, and therefore which GPU is the better fit. This project measures that empirically for real CNN training runs.

## What it does

- Profiles training of two CNN architectures — **ResNet-18** and **AlexNet** — on two NVIDIA data-center GPUs, **A100** and **V100** (NYU HPC).
- Measures floating-point work and memory traffic directly from hardware counters.
- Computes **arithmetic intensity**, **attainable FLOP/s**, and runtime for each (model, GPU) pair.
- Places each result on a Roofline plot and discusses the compute- vs. memory-bound behavior.

## Method

The training harness (`main.py`) is a standard PyTorch ImageNet training loop over `torchvision` models. To isolate GPU behavior, it trains on PyTorch **FakeData** (randomly generated, same input dimensions as ImageNet) for a short run rather than the full dataset, so results reflect hardware performance rather than data pipeline or convergence.

The region of interest in the training loop is bracketed with `torch.cuda.profiler` start/stop, and the script is run under **NVIDIA Nsight Compute (`ncu`)** to collect SASS-level FP instruction counts and DRAM read/write metrics:

```bash
ncu -f --log-file resnet18_A100.log --profile-from-start off --replay-mode application \
    --metrics smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__sectors_write.sum,dram__bytes_write.sum.per_second,\
dram__sectors_read.sum,dram__bytes_read.sum.per_second \
    --target-processes all python3 ./main.py --batch-size 4 --epochs 1 --print-freq 10 -a resnet18 --dummy
```

Derived quantities:

- **FLOPs** = `fadd` + `fmul` + 2 × `ffma`
- **Bytes accessed** = DRAM read + write traffic (from sector/byte counters)
- **Arithmetic intensity** = FLOPs / bytes
- **Attainable performance** = FLOPs / total kernel time

## Results (summary)

Arithmetic intensity and achieved throughput were computed for each architecture on each GPU and plotted against each device's Roofline. Representative measurements:

| Model     | GPU  | Arithmetic intensity (FLOPs/byte) | Performance |
|-----------|------|-----------------------------------|-------------|
| ResNet-18 | V100 | ~23.5                             | ~4.60 TFLOPS |
| ResNet-18 | A100 | ~0.55                             | ~0.12 TFLOPS |
| AlexNet   | V100 | ~10.4                             | ~3.54 TFLOPS |
| AlexNet   | A100 | ~5.24                             | ~1.78 TFLOPS |

The full methodology, per-counter measurements, and Roofline plots are in **`Roofline Modeling Report.pdf`**.

## Repository structure

```
main.py                      # PyTorch training + profiling harness
Roofline Modeling Report.pdf # full write-up: method, measurements, plots
README.md
```

## Tech stack

Python · PyTorch · NVIDIA Nsight Compute (ncu) · CUDA GPUs (A100 / V100) · NYU HPC

## Author

Mihir Prajapati — NYU, Cloud & Machine Learning (Spring 2024).
