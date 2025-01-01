```
==================================================
Analysis for trajectory_balance_loss
==================================================

Summary Statistics:

Early stage:
  Mean   : 7.43026
  Std    : 2.89578
  Median : 8.67855

Mid stage:
  Mean   : 1.72514
  Std    : 2.01813
  Median : 0.67939

Late stage:
  Mean   : 0.00124271
  Std    : 0.002363
  Median : 8.49505e-05

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 4.11994e-47
  Significant  : Yes
  Effect size  : -2.28586
  Mean diff    : -5.70512
  Median diff  : -7.99916

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 7.77444e-84
  Significant  : Yes
  Effect size  : -1.20803
  Mean diff    : -1.7239
  Median diff  : -0.679305

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 9.29597e-48
  Significant  : Yes
  Effect size  : -3.62812
  Mean diff    : -7.42902
  Median diff  : -8.67846

==================================================
Analysis for terminal_reward
==================================================

Summary Statistics:

Early stage:
  Mean   : 35.9231
  Std    : 8.40188
  Median : 36

Mid stage:
  Mean   : 43.5121
  Std    : 9.622
  Median : 43

Late stage:
  Mean   : 49.098
  Std    : 7.93491
  Median : 49

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 5.72986e-18
  Significant  : Yes
  Effect size  : 0.840184
  Mean diff    : 7.58902
  Median diff  : 7

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 8.90974e-12
  Significant  : Yes
  Effect size  : 0.633407
  Mean diff    : 5.58594
  Median diff  : 6

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 1.10968e-46
  Significant  : Yes
  Effect size  : 1.61226
  Mean diff    : 13.175
  Median diff  : 13

==================================================
Analysis for exploration_ratio
==================================================

Summary Statistics:

Early stage:
  Mean   : 0.992501
  Std    : 0.0482797
  Median : 1

Mid stage:
  Mean   : 1
  Std    : 0
  Median : 1

Late stage:
  Mean   : 1
  Std    : 0
  Median : 1

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U (due to uniform data)
  P-value      : 0.00245216
  Significant  : Yes
  Effect size  : 0.219652
  Mean diff    : 0.00749868
  Median diff  : 0

MID-LATE:
  Test used    : Mann-Whitney U (due to uniform data)
  P-value      : 1
  Significant  : No
  Effect size  : 0
  Mean diff    : 0
  Median diff  : 0

EARLY-LATE:
  Test used    : Mann-Whitney U (due to uniform data)
  P-value      : 0.00213111
  Significant  : Yes
  Effect size  : 0.219652
  Mean diff    : 0.00749868
  Median diff  : 0

==================================================
Analysis for forward_entropy
==================================================

Summary Statistics:

Early stage:
  Mean   : 0.0910028
  Std    : 0.000460817
  Median : 0.0911968

Mid stage:
  Mean   : 0.0850012
  Std    : 0.00345056
  Median : 0.0855755

Late stage:
  Mean   : 0.0769377
  Std    : 0.000574968
  Median : 0.0766502

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 1.39375e-82
  Significant  : Yes
  Effect size  : -2.4381
  Mean diff    : -0.00600162
  Median diff  : -0.00562132

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 7.50099e-84
  Significant  : Yes
  Effect size  : -3.25985
  Mean diff    : -0.00806349
  Median diff  : -0.00892528

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 1.05232e-83
  Significant  : Yes
  Effect size  : -26.9899
  Mean diff    : -0.0140651
  Median diff  : -0.0145466

==================================================
Analysis for backward_entropy
==================================================

Summary Statistics:

Early stage:
  Mean   : 0.0592377
  Std    : 0.00487484
  Median : 0.0616291

Mid stage:
  Mean   : 0.0163421
  Std    : 0.0117586
  Median : 0.0109835

Late stage:
  Mean   : 0.00542834
  Std    : 0.000253375
  Median : 0.00539537

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 1.46542e-82
  Significant  : Yes
  Effect size  : -4.76574
  Mean diff    : -0.0428955
  Median diff  : -0.0506456

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 3.26299e-78
  Significant  : Yes
  Effect size  : -1.3123
  Mean diff    : -0.0109138
  Median diff  : -0.00558811

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 1.10488e-83
  Significant  : Yes
  Effect size  : -15.5892
  Mean diff    : -0.0538093
  Median diff  : -0.0562337
```
