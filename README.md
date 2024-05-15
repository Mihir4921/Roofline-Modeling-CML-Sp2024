# Roofline-Modeling-CML-Sp2024
Perform Roofline Modeling on different GPUs

Run the following code in HPC shell to perform GPU profiling.

```
ncu -f --log-file resnet18_A100.log --profile-from-start off --replay-mode application --metrics smsp_sass_thread_inst_executed_op_fp16_pred_on.sum,smspsass_thread_inst_executed_op_fadd_pre d_on.sum,smspsass_thread_inst_executed_op_fmul_pred_on.sum,smspsass_thread_inst_executed_op _ffma_pred_on.sum,dramsectors_write.sum,drambytes_write.sum.per_second,dramsectors_read.sum,d ram_bytes_read.sum.per_second --target-processes all python3 ./main.py --batch-size 4 --epochs 1 -- print-freq 10 -a resnet18 --dummy
```
