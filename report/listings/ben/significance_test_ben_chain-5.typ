```
==================================================
Analysis for Q-Learning Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 9.0621e+09
  Std      : 1.11129e+11
  Median   : 359028

Mid stage:
  Mean     : 51210.2
  Std      : 82246.9
  Median   : 24168.2

Late stage:
  Mean     : 3529.9
  Std      : 5630.64
  Median   : 1565.3

Statistical Tests:

EARLY-MID:
  Test used     : Mann-Whitney U
  P-value       : 1.61872e-55
  Significant   : Yes
  Effect size   : -0.115322
  Mean diff     : -9.06205e+09
  Median diff   : -334860

MID-LATE:
  Test used     : Mann-Whitney U
  P-value       : 1.27667e-59
  Significant   : Yes
  Effect size   : -0.817936
  Mean diff     : -47680.3
  Median diff   : -22602.9

EARLY-LATE:
  Test used     : Mann-Whitney U
  P-value       : 1.30966e-94
  Significant   : Yes
  Effect size   : -0.115323
  Mean diff     : -9.0621e+09
  Median diff   : -357463

==================================================
Analysis for Epistemic Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 2214.01
  Std      : 1351.49
  Median   : 1759.53

Mid stage:
  Mean     : 915.499
  Std      : 561.958
  Median   : 724.881

Late stage:
  Mean     : 605.353
  Std      : 3316.38
  Median   : 288.135

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 1.72322e-180
  Significant   : Yes
  Effect size   : -1.25464
  Mean diff     : -1298.51
  Median diff   : -1034.65

MID-LATE:
  Test used     : Mann-Whitney U
  P-value       : 2.25913e-191
  Significant   : Yes
  Effect size   : -0.130398
  Mean diff     : -310.147
  Median diff   : -436.746

EARLY-LATE:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : -0.635261
  Mean diff     : -1608.66
  Median diff   : -1471.4

==================================================
Analysis for Rewards
==================================================

Summary Statistics:

Early stage:
  Mean     : -109.125
  Std      : 120.27
  Median   : -86

Mid stage:
  Mean     : -113.824
  Std      : 104.901
  Median   : -99

Late stage:
  Mean     : -96.8824
  Std      : 85.1993
  Median   : -105

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 0.908325
  Significant   : No
  Effect size   : -0.0416361
  Mean diff     : -4.69853
  Median diff   : -13

MID-LATE:
  Test used     : t-test
  P-value       : 0.619497
  Significant   : No
  Effect size   : 0.177284
  Mean diff     : 16.9412
  Median diff   : -6

EARLY-LATE:
  Test used     : t-test
  P-value       : 0.744724
  Significant   : No
  Effect size   : 0.117469
  Mean diff     : 12.2426
  Median diff   : -19

==================================================
Analysis for Cumulative Returns
==================================================

Summary Statistics:

Early stage:
  Mean     : -1084.31
  Std      : 438.654
  Median   : -1081.5

Mid stage:
  Mean     : -2960.94
  Std      : 548.241
  Median   : -3043

Late stage:
  Mean     : -4479.41
  Std      : 467.605
  Median   : -4589

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 1.02639e-11
  Significant   : Yes
  Effect size   : -3.77986
  Mean diff     : -1876.63
  Median diff   : -1961.5

MID-LATE:
  Test used     : t-test
  P-value       : 1.24336e-09
  Significant   : Yes
  Effect size   : -2.98019
  Mean diff     : -1518.47
  Median diff   : -1546

EARLY-LATE:
  Test used     : t-test
  P-value       : 8.83529e-20
  Significant   : Yes
  Effect size   : -7.48874
  Mean diff     : -3395.1
  Median diff   : -3507.5
```
