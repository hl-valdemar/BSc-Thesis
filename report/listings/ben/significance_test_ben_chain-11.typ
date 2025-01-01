```
==================================================
Analysis for Q-Learning Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 150.151
  Std      : 610.171
  Median   : 2.32373

Mid stage:
  Mean     : 1.27351
  Std      : 0.58875
  Median   : 1.13778

Late stage:
  Mean     : 1.09848
  Std      : 0.48181
  Median   : 1.00344

Statistical Tests:

EARLY-MID:
  Test used     : Mann-Whitney U
  P-value       : 7.15432e-75
  Significant   : Yes
  Effect size   : -0.345059
  Mean diff     : -148.878
  Median diff   : -1.18596

MID-LATE:
  Test used     : t-test
  P-value       : 1.97005e-08
  Significant   : Yes
  Effect size   : -0.325374
  Mean diff     : -0.175033
  Median diff   : -0.134332

EARLY-LATE:
  Test used     : Mann-Whitney U
  P-value       : 4.34471e-101
  Significant   : Yes
  Effect size   : -0.345464
  Mean diff     : -149.053
  Median diff   : -1.32029

==================================================
Analysis for Epistemic Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 9005.47
  Std      : 424831
  Median   : 230.341

Mid stage:
  Mean     : 1946.32
  Std      : 53466.3
  Median   : -11.3205

Late stage:
  Mean     : 295.921
  Std      : 9372.83
  Median   : -38.3282

Statistical Tests:

EARLY-MID:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : -0.0233152
  Mean diff     : -7059.15
  Median diff   : -241.661

MID-LATE:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : -0.0429982
  Mean diff     : -1650.4
  Median diff   : -27.0076

EARLY-LATE:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : -0.028986
  Mean diff     : -8709.55
  Median diff   : -268.669

==================================================
Analysis for Rewards
==================================================

Summary Statistics:

Early stage:
  Mean     : -82.875
  Std      : 76.4566
  Median   : -86

Mid stage:
  Mean     : -144.471
  Std      : 85.5083
  Median   : -137

Late stage:
  Mean     : -75.6471
  Std      : 84.6633
  Median   : -67

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 0.0430657
  Significant   : Yes
  Effect size   : -0.759419
  Mean diff     : -61.5956
  Median diff   : -51

MID-LATE:
  Test used     : t-test
  P-value       : 0.0289021
  Significant   : Yes
  Effect size   : 0.808862
  Mean diff     : 68.8235
  Median diff   : 70

EARLY-LATE:
  Test used     : t-test
  P-value       : 0.80504
  Significant   : No
  Effect size   : 0.0896051
  Mean diff     : 7.22794
  Median diff   : 19

==================================================
Analysis for Cumulative Returns
==================================================

Summary Statistics:

Early stage:
  Mean     : -598.438
  Std      : 432.208
  Median   : -680.5

Mid stage:
  Mean     : -2539.53
  Std      : 620.485
  Median   : -2400

Late stage:
  Mean     : -4302.35
  Std      : 429.992
  Median   : -4042

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 2.8519e-11
  Significant   : Yes
  Effect size   : -3.63025
  Mean diff     : -1941.09
  Median diff   : -1719.5

MID-LATE:
  Test used     : t-test
  P-value       : 1.17108e-10
  Significant   : Yes
  Effect size   : -3.30237
  Mean diff     : -1762.82
  Median diff   : -1642

EARLY-LATE:
  Test used     : t-test
  P-value       : 1.54372e-21
  Significant   : Yes
  Effect size   : -8.59174
  Mean diff     : -3703.92
  Median diff   : -3361.5
```
