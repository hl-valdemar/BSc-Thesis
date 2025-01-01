```
==================================================
Analysis for Q-Learning Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 2085.95
  Std      : 2162.4
  Median   : 1373.73

Mid stage:
  Mean     : 576.122
  Std      : 649.869
  Median   : 338.179

Late stage:
  Mean     : 156.021
  Std      : 191.834
  Median   : 67.8597

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 1.61151e-36
  Significant   : Yes
  Effect size   : -0.94565
  Mean diff     : -1509.83
  Median diff   : -1035.55

MID-LATE:
  Test used     : t-test
  P-value       : 7.51887e-33
  Significant   : Yes
  Effect size   : -0.876801
  Mean diff     : -420.101
  Median diff   : -270.319

EARLY-LATE:
  Test used     : t-test
  P-value       : 1.0056e-60
  Significant   : Yes
  Effect size   : -1.25724
  Mean diff     : -1929.93
  Median diff   : -1305.87

==================================================
Analysis for Epistemic Loss
==================================================

Summary Statistics:

Early stage:
  Mean     : 1376.98
  Std      : 1241.08
  Median   : 988.406

Mid stage:
  Mean     : 1258.87
  Std      : 31145.6
  Median   : 161.888

Late stage:
  Mean     : 2677.52
  Std      : 60170.7
  Median   : 5.31501

Statistical Tests:

EARLY-MID:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : -0.00535891
  Mean diff     : -118.114
  Median diff   : -826.518

MID-LATE:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : 0.0296113
  Mean diff     : 1418.65
  Median diff   : -156.573

EARLY-LATE:
  Test used     : Mann-Whitney U
  P-value       : 0
  Significant   : Yes
  Effect size   : 0.0305605
  Mean diff     : 1300.54
  Median diff   : -983.091

==================================================
Analysis for Rewards
==================================================

Summary Statistics:

Early stage:
  Mean     : -80.875
  Std      : 70.1827
  Median   : -57.5

Mid stage:
  Mean     : -114.882
  Std      : 97.6795
  Median   : -137

Late stage:
  Mean     : -79.6471
  Std      : 75.2853
  Median   : -86

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 0.276686
  Significant   : No
  Effect size   : -0.399853
  Mean diff     : -34.0074
  Median diff   : -79.5

MID-LATE:
  Test used     : t-test
  P-value       : 0.261584
  Significant   : No
  Effect size   : 0.404055
  Mean diff     : 35.2353
  Median diff   : 51

EARLY-LATE:
  Test used     : t-test
  P-value       : 0.962895
  Significant   : No
  Effect size   : 0.0168723
  Mean diff     : 1.22794
  Median diff   : -28.5

==================================================
Analysis for Cumulative Returns
==================================================

Summary Statistics:

Early stage:
  Mean     : -756.438
  Std      : 448.425
  Median   : -829.5

Mid stage:
  Mean     : -2396.41
  Std      : 540.392
  Median   : -2349

Late stage:
  Mean     : -4111.24
  Std      : 491.61
  Median   : -4231

Statistical Tests:

EARLY-MID:
  Test used     : t-test
  P-value       : 2.4656e-10
  Significant   : Yes
  Effect size   : -3.30279
  Mean diff     : -1639.97
  Median diff   : -1519.5

MID-LATE:
  Test used     : t-test
  P-value       : 1.03534e-10
  Significant   : Yes
  Effect size   : -3.31959
  Mean diff     : -1714.82
  Median diff   : -1882

EARLY-LATE:
  Test used     : t-test
  P-value       : 3.69339e-19
  Significant   : Yes
  Effect size   : -7.13008
  Mean diff     : -3354.8
  Median diff   : -3401.5
```
