```
==================================================
Analysis for Q-Learning Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 201.363
  Std      : 646.288
  Median   : 28.6683

Mid stage:
  Mean     : 7.60222
  Std      : 6.14727
  Median   : 6.00191

Late stage:
  Mean     : 3.53114
  Std      : 2.27404
  Median   : 2.88227

Statistical Tests:

EARLY-MID:
  Test used     : Mann-Whitney U
  P-value       : 3.91253e-87
  Significant   : Yes
  Effect size   : -0.423969
  Mean diff     : -193.76
  Median diff   : -22.6664

MID-LATE:
  Test used     : t-test
  P-value       : 9.57251e-41
  Significant   : Yes
  Effect size   : -0.878398
  Mean diff     : -4.07108
  Median diff   : -3.11963

EARLY-LATE:
  Test used     : Mann-Whitney U
  P-value       : 7.90234e-134
  Significant   : Yes
  Effect size   : -0.432894
  Mean diff     : -197.831
  Median diff   : -25.7861

==================================================
Analysis for Epistemic Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 854.156
  Std      : 1244.76
  Median   : 476.503

Mid stage:
  Mean     : 2051.78
  Std      : 109092
  Median   : 9.81774

Late stage:
  Mean     : -21.3259
  Std      : 417.914
  Median   : -29.5177

Statistical Tests:

EARLY-MID:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : 0.0155244
  Mean diff     : 1197.62
  Median diff   : -466.685

MID-LATE:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : -0.0268745
  Mean diff     : -2073.1
  Median diff   : -39.3355

EARLY-LATE:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : -0.942936
  Mean diff     : -875.482
  Median diff   : -506.021

==================================================
Analysis for Rewards
==================================================

Summary Statistics:

Early stage:
  Mean     : -139.375
  Std      : 107.425
  Median   : -111.5

Mid stage:
  Mean     : -131
  Std      : 83.4245
  Median   : -118

Late stage:
  Mean     : -128.353
  Std      : 74.1294
  Median   : -137

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 0.809435
  Significant   : No
  Effect size   : 0.0870797
  Mean diff     : 8.375
  Median diff   : -6.5

MID-LATE:
  Test used     : t-test
  P-value       : 0.925005
  Significant   : No
  Effect size   : 0.0335436
  Mean diff     : 2.64706
  Median diff   : -19

EARLY-LATE:
  Test used     : t-test
  P-value       : 0.740539
  Significant   : No
  Effect size   : 0.119427
  Mean diff     : 11.0221
  Median diff   : -25.5

==================================================
Analysis for Cumulative Returns
==================================================

Summary Statistics:

Early stage:
  Mean     : -1496.19
  Std      : 584.326
  Median   : -1587.5

Mid stage:
  Mean     : -3164.82
  Std      : 677.043
  Median   : -2960

Late stage:
  Mean     : -5642.65
  Std      : 604.554
  Median   : -5607

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 3.02444e-08
  Significant   : Yes
  Effect size   : -2.63863
  Mean diff     : -1668.64
  Median diff   : -1372.5

MID-LATE:
  Test used     : t-test
  P-value       : 2.52487e-12
  Significant   : Yes
  Effect size   : -3.8606
  Mean diff     : -2477.82
  Median diff   : -2647

EARLY-LATE:
  Test used     : t-test
  P-value       : 6.79593e-19
  Significant   : Yes
  Effect size   : -6.97439
  Mean diff     : -4146.46
  Median diff   : -4019.5
```
