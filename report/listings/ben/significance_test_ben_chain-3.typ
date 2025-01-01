```
==================================================
Analysis for Q-Learning Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 2418.4
  Std      : 2948.42
  Median   : 1382.44

Mid stage:
  Mean     : 850.307
  Std      : 865.315
  Median   : 548.499

Late stage:
  Mean     : 249.975
  Std      : 267.485
  Median   : 164.514

Statistical Tests:

EARLY-MID:
  Test used     : Mann-Whitney U
  P-value       : 2.75096e-12
  Significant   : Yes
  Effect size   : -0.721698
  Mean diff     : -1568.09
  Median diff   : -833.938

MID-LATE:
  Test used     : t-test
  P-value       : 3.35826e-19
  Significant   : Yes
  Effect size   : -0.937378
  Mean diff     : -600.332
  Median diff   : -383.985

EARLY-LATE:
  Test used     : Mann-Whitney U
  P-value       : 2.16325e-39
  Significant   : Yes
  Effect size   : -1.03583
  Mean diff     : -2168.42
  Median diff   : -1217.92

==================================================
Analysis for Epistemic Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 3738.12
  Std      : 4340.37
  Median   : 3595.12

Mid stage:
  Mean     : 2224.3
  Std      : 926.447
  Median   : 1906.22

Late stage:
  Mean     : 1480.38
  Std      : 614.314
  Median   : 1450.01

Statistical Tests:

EARLY-MID:
  Test used     : Mann-Whitney U
  P-value       : 1.87536e-39
  Significant   : Yes
  Effect size   : -0.482378
  Mean diff     : -1513.82
  Median diff   : -1688.89

MID-LATE:
  Test used     : t-test
  P-value       : 3.41227e-46
  Significant   : Yes
  Effect size   : -0.946429
  Mean diff     : -743.922
  Median diff   : -456.211

EARLY-LATE:
  Test used     : Mann-Whitney U
  P-value       : 1.97483e-98
  Significant   : Yes
  Effect size   : -0.728375
  Mean diff     : -2257.74
  Median diff   : -2145.11

==================================================
Analysis for Rewards
==================================================

Summary Statistics:

Early stage:
  Mean     : -81.25
  Std      : 103.5
  Median   : -86

Mid stage:
  Mean     : -116.765
  Std      : 95.9329
  Median   : -99

Late stage:
  Mean     : -75.1765
  Std      : 90.9287
  Median   : -73

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 0.329132
  Significant   : No
  Effect size   : -0.355901
  Mean diff     : -35.5147
  Median diff   : -13

MID-LATE:
  Test used     : t-test
  P-value       : 0.217299
  Significant   : No
  Effect size   : 0.444964
  Mean diff     : 41.5882
  Median diff   : 26

EARLY-LATE:
  Test used     : t-test
  P-value       : 0.863135
  Significant   : No
  Effect size   : 0.0623456
  Mean diff     : 6.07353
  Median diff   : 13

==================================================
Analysis for Cumulative Returns
==================================================

Summary Statistics:

Early stage:
  Mean     : -572.312
  Std      : 353.307
  Median   : -648.5

Mid stage:
  Mean     : -2356.35
  Std      : 487.332
  Median   : -2300

Late stage:
  Mean     : -3809.12
  Std      : 401.997
  Median   : -3671

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 8.10375e-13
  Significant   : Yes
  Effect size   : -4.19155
  Mean diff     : -1784.04
  Median diff   : -1651.5

MID-LATE:
  Test used     : t-test
  P-value       : 1.68029e-10
  Significant   : Yes
  Effect size   : -3.25217
  Mean diff     : -1452.76
  Median diff   : -1371

EARLY-LATE:
  Test used     : t-test
  P-value       : 1.87071e-21
  Significant   : Yes
  Effect size   : -8.55312
  Mean diff     : -3236.81
  Median diff   : -3022.5
```
