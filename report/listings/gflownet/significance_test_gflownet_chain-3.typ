```
==================================================
Analysis for trajectory_balance_loss
==================================================

Summary Statistics:

Early stage:
  Mean   : 6.82581
  Std    : 2.81166
  Median : 8.15058

Mid stage:
  Mean   : 0.931558
  Std    : 1.26559
  Median : 0.262708

Late stage:
  Mean   : 0.000422112
  Std    : 0.000892446
  Median : 1.66286e-05

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 3.9726e-47
  Significant  : Yes
  Effect size  : -2.70345
  Mean diff    : -5.89425
  Median diff  : -7.88788

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 7.59106e-84
  Significant  : Yes
  Effect size  : -1.04048
  Mean diff    : -0.931136
  Median diff  : -0.262691

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 9.29597e-48
  Significant  : Yes
  Effect size  : -3.43304
  Mean diff    : -6.82538
  Median diff  : -8.15057

==================================================
Analysis for terminal_reward
==================================================

Summary Statistics:

Early stage:
  Mean   : 35.3644
  Std    : 9.19597
  Median : 36

Mid stage:
  Mean   : 44.1653
  Std    : 9.85363
  Median : 43

Late stage:
  Mean   : 49.2667
  Std    : 7.42465
  Median : 49

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 4.4844e-21
  Significant  : Yes
  Effect size  : 0.923453
  Mean diff    : 8.80095
  Median diff  : 7

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 4.06779e-10
  Significant  : Yes
  Effect size  : 0.584742
  Mean diff    : 5.10134
  Median diff  : 6

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 6.93379e-48
  Significant  : Yes
  Effect size  : 1.66348
  Mean diff    : 13.9023
  Median diff  : 13

==================================================
Analysis for exploration_ratio
==================================================

Summary Statistics:

Early stage:
  Mean   : 0.992713
  Std    : 0.0505792
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
  P-value      : 0.00433248
  Significant  : Yes
  Effect size  : 0.20376
  Mean diff    : 0.00728745
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
  P-value      : 0.00382047
  Significant  : Yes
  Effect size  : 0.20376
  Mean diff    : 0.00728745
  Median diff  : 0

==================================================
Analysis for forward_entropy
==================================================

Summary Statistics:

Early stage:
  Mean   : 0.272931
  Std    : 0.00158218
  Median : 0.27355

Mid stage:
  Mean   : 0.248951
  Std    : 0.0136987
  Median : 0.24999

Late stage:
  Mean   : 0.223683
  Std    : 0.00105377
  Median : 0.223144

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 1.39375e-82
  Significant  : Yes
  Effect size  : -2.45926
  Mean diff    : -0.0239799
  Median diff  : -0.0235602

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 7.50099e-84
  Significant  : Yes
  Effect size  : -2.60084
  Mean diff    : -0.0252673
  Median diff  : -0.0268458

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 1.05232e-83
  Significant  : Yes
  Effect size  : -36.6358
  Mean diff    : -0.0492472
  Median diff  : -0.0504061

==================================================
Analysis for backward_entropy
==================================================

Summary Statistics:

Early stage:
  Mean   : 0.20616
  Std    : 0.0309031
  Median : 0.221136

Mid stage:
  Mean   : 0.0424959
  Std    : 0.0271554
  Median : 0.0293473

Late stage:
  Mean   : 0.0199545
  Std    : 0.00105893
  Median : 0.0197963

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 1.46542e-82
  Significant  : Yes
  Effect size  : -5.62619
  Mean diff    : -0.163664
  Median diff  : -0.191789

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 1.65605e-67
  Significant  : Yes
  Effect size  : -1.17304
  Mean diff    : -0.0225415
  Median diff  : -0.00955104

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 1.10488e-83
  Significant  : Yes
  Effect size  : -8.5163
  Mean diff    : -0.186205
  Median diff  : -0.20134
```
