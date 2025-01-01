```
==================================================
Analysis for trajectory_balance_loss
==================================================

Summary Statistics:

Early stage:
  Mean   : 7.27744
  Std    : 2.84408
  Median : 8.48092

Mid stage:
  Mean   : 1.74531
  Std    : 1.98533
  Median : 0.719583

Late stage:
  Mean   : 0.00267701
  Std    : 0.00513376
  Median : 0.000157041

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 4.59541e-47
  Significant  : Yes
  Effect size  : -2.25563
  Mean diff    : -5.53212
  Median diff  : -7.76134

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 8.15457e-84
  Significant  : Yes
  Effect size  : -1.24133
  Mean diff    : -1.74264
  Median diff  : -0.719426

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 9.29597e-48
  Significant  : Yes
  Effect size  : -3.61736
  Mean diff    : -7.27476
  Median diff  : -8.48077

==================================================
Analysis for terminal_reward
==================================================

Summary Statistics:

Early stage:
  Mean   : 34.081
  Std    : 8.16854
  Median : 33

Mid stage:
  Mean   : 41.0726
  Std    : 9.20453
  Median : 42

Late stage:
  Mean   : 50.0353
  Std    : 8.54646
  Median : 49

Statistical Tests:

EARLY-MID:
  Test used    : t-test
  P-value      : 8.03004e-18
  Significant  : Yes
  Effect size  : 0.803451
  Mean diff    : 6.99161
  Median diff  : 9

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 2.01281e-24
  Significant  : Yes
  Effect size  : 1.00913
  Mean diff    : 8.96271
  Median diff  : 7

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 1.2424e-57
  Significant  : Yes
  Effect size  : 1.90849
  Mean diff    : 15.9543
  Median diff  : 16

==================================================
Analysis for exploration_ratio
==================================================

Summary Statistics:

Early stage:
  Mean   : 0.993288
  Std    : 0.0459779
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
  P-value      : 0.00139134
  Significant  : Yes
  Effect size  : 0.206455
  Mean diff    : 0.00671212
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
  P-value      : 0.0011917
  Significant  : Yes
  Effect size  : 0.206455
  Mean diff    : 0.00671212
  Median diff  : 0

==================================================
Analysis for forward_entropy
==================================================

Summary Statistics:

Early stage:
  Mean   : 0.109361
  Std    : 0.000378315
  Median : 0.109516

Mid stage:
  Mean   : 0.102774
  Std    : 0.00446085
  Median : 0.103877

Late stage:
  Mean   : 0.0905926
  Std    : 0.00113378
  Median : 0.0900322

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 1.39375e-82
  Significant  : Yes
  Effect size  : -2.08054
  Mean diff    : -0.00658623
  Median diff  : -0.00563877

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 7.50099e-84
  Significant  : Yes
  Effect size  : -3.74293
  Mean diff    : -0.0121817
  Median diff  : -0.0138448

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 1.05232e-83
  Significant  : Yes
  Effect size  : -22.2049
  Mean diff    : -0.018768
  Median diff  : -0.0194836

==================================================
Analysis for backward_entropy
==================================================

Summary Statistics:

Early stage:
  Mean   : 0.072528
  Std    : 0.00566449
  Median : 0.0752685

Mid stage:
  Mean   : 0.0205243
  Std    : 0.0146301
  Median : 0.0136152

Late stage:
  Mean   : 0.00679753
  Std    : 0.000457729
  Median : 0.00675402

Statistical Tests:

EARLY-MID:
  Test used    : Mann-Whitney U
  P-value      : 1.53827e-82
  Significant  : Yes
  Effect size  : -4.6878
  Mean diff    : -0.0520037
  Median diff  : -0.0616532

MID-LATE:
  Test used    : Mann-Whitney U
  P-value      : 2.35375e-77
  Significant  : Yes
  Effect size  : -1.32624
  Mean diff    : -0.0137268
  Median diff  : -0.00686123

EARLY-LATE:
  Test used    : Mann-Whitney U
  P-value      : 1.10488e-83
  Significant  : Yes
  Effect size  : -16.3571
  Mean diff    : -0.0657304
  Median diff  : -0.0685144
```
