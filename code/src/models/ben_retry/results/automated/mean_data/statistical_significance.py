import json
import warnings
from typing import Dict, Tuple

import numpy as np
from scipy import stats


def load_and_prepare_data(file_path: str) -> Dict:
    """Load the JSON data and prepare it for analysis."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def split_into_stages(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into early, mid, and late stages."""
    total_steps = len(data)
    early_end = int(total_steps * 0.33)
    mid_end = int(total_steps * 0.66)

    early = data[:early_end]
    mid = data[early_end:mid_end]
    late = data[mid_end:]

    return early, mid, late


def calculate_robust_effect_size(stage1: np.ndarray, stage2: np.ndarray) -> float:
    """Calculate effect size with robustness to near-zero variances."""
    diff = np.mean(stage2) - np.mean(stage1)
    # Use pooled standard deviation with regularization
    var1, var2 = np.var(stage1), np.var(stage2)
    pooled_std = np.sqrt((var1 + var2) / 2 + 1e-10)

    return diff / pooled_std if pooled_std > 0 else np.inf * np.sign(diff)


def perform_statistical_tests(
    early: np.ndarray, mid: np.ndarray, late: np.ndarray, metric_name: str
) -> Dict:
    """Perform statistical tests between stages for a given metric."""
    results = {}
    stages = [
        ("early-mid", early, mid),
        ("mid-late", mid, late),
        ("early-late", early, late),
    ]

    for stage_name, stage1, stage2 in stages:
        # For nearly identical data, use non-parametric test by default
        if np.std(stage1) < 1e-6 or np.std(stage2) < 1e-6:
            stat, p_val = stats.mannwhitneyu(stage1, stage2, alternative="two-sided")
            test_name = "Mann-Whitney U (due to uniform data)"
        else:
            # Perform multiple tests
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Kolmogorov-Smirnov test for distribution differences
                ks_stat, ks_p = stats.ks_2samp(stage1, stage2)

                # Mann-Whitney U test for median differences
                mw_stat, mw_p = stats.mannwhitneyu(
                    stage1, stage2, alternative="two-sided"
                )

                # t-test for mean differences (if approximately normal)
                t_stat, t_p = stats.ttest_ind(stage1, stage2)

                # Choose most appropriate test based on data characteristics
                if np.abs(stats.skew(stage1)) < 2 and np.abs(stats.skew(stage2)) < 2:
                    stat, p_val = t_stat, t_p
                    test_name = "t-test"
                else:
                    stat, p_val = mw_stat, mw_p
                    test_name = "Mann-Whitney U"

                # Include KS test results separately
                ks_result = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "significant": ks_p < 0.05,
                }

        effect_size = calculate_robust_effect_size(stage1, stage2)

        # Calculate additional statistical measures
        relative_change = ((np.mean(stage2) - np.mean(stage1)) / np.mean(stage1)) * 100
        variance_ratio = np.var(stage2) / np.var(stage1)

        results[stage_name] = {
            "test_used": test_name,
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
            "mean_diff": float(np.mean(stage2) - np.mean(stage1)),
            "median_diff": float(np.median(stage2) - np.median(stage1)),
            "effect_size": float(effect_size),
            "relative_change_percent": float(relative_change),
            "variance_ratio": float(variance_ratio),
            "kolmogorov_smirnov_test": ks_result,
        }

    return results


def analyze_ben_training(file_path: str) -> Dict:
    """Main analysis function for BEN training data."""
    # Load data
    data = load_and_prepare_data(file_path)

    # Define metrics to analyze
    metrics = {
        "Q-Learning Loss": data["losses_q"],
        "Epistemic Loss": data["losses_epistemic"],
        "Rewards": data["rewards"],
        "Branch Choices": data["branches"],
        "Cumulative Returns": data["cumulative_returns"],
    }

    # Results container
    results = {}

    # Analyze each metric
    for metric_name, metric_data in metrics.items():
        # Split into stages
        early, mid, late = split_into_stages(np.array(metric_data))

        # Calculate expanded summary statistics
        summary_stats = {
            "early": {
                "mean": float(np.mean(early)),
                "std": float(np.std(early)),
                "median": float(np.median(early)),
                "skewness": float(stats.skew(early)),
                "kurtosis": float(stats.kurtosis(early)),
                "q25": float(np.percentile(early, 25)),
                "q75": float(np.percentile(early, 75)),
            },
            "mid": {
                "mean": float(np.mean(mid)),
                "std": float(np.std(mid)),
                "median": float(np.median(mid)),
                "skewness": float(stats.skew(mid)),
                "kurtosis": float(stats.kurtosis(mid)),
                "q25": float(np.percentile(mid, 25)),
                "q75": float(np.percentile(mid, 75)),
            },
            "late": {
                "mean": float(np.mean(late)),
                "std": float(np.std(late)),
                "median": float(np.median(late)),
                "skewness": float(stats.skew(late)),
                "kurtosis": float(stats.kurtosis(late)),
                "q25": float(np.percentile(late, 25)),
                "q75": float(np.percentile(late, 75)),
            },
        }

        # Perform statistical tests
        test_results = perform_statistical_tests(early, mid, late, metric_name)

        results[metric_name] = {
            "summary_statistics": summary_stats,
            "statistical_tests": test_results,
        }

    return results


def print_analysis_results(results: Dict):
    """Print analysis results in a clear, formatted way."""
    for metric_name, metric_results in results.items():
        print(f"\n{'='*50}")
        print(f"Analysis for {metric_name}")
        print(f"{'='*50}")

        # Print summary statistics
        print("\nSummary Statistics:")
        for stage in ["early", "mid", "late"]:
            stats = metric_results["summary_statistics"][stage]
            print(f"\n{stage.capitalize()} stage:")
            print(f"  Mean     : {stats['mean']:.6g}")
            print(f"  Std      : {stats['std']:.6g}")
            print(f"  Median   : {stats['median']:.6g}")
            # print(f"  Skewness : {stats['skewness']:.6g}")
            # print(f"  Q25-Q75  : [{stats['q25']:.6g}, {stats['q75']:.6g}]")

        # Print statistical test results
        print("\nStatistical Tests:")
        for comparison, test_results in metric_results["statistical_tests"].items():
            print(f"\n{comparison.upper()}:")
            print(f"  Test used     : {test_results['test_used']}")
            print(f"  P-value       : {test_results['p_value']:.6g}")
            print(f"  Significant   : {'Yes' if test_results['significant'] else 'No'}")
            print(f"  Effect size   : {test_results['effect_size']:.6g}")
            print(f"  Mean diff     : {test_results['mean_diff']:.6g}")
            print(f"  Median diff   : {test_results['median_diff']:.6g}")
            # print(f"  % Change      : {test_results['relative_change_percent']:.2f}%")
            # print(f"  Variance ratio: {test_results['variance_ratio']:.6g}")

            # Print KS test results
            # ks = test_results["kolmogorov_smirnov_test"]
            # print(
            #     f"  KS test       : p = {ks['p_value']:.6g} "
            #     f"({'significant' if ks['significant'] else 'not significant'})"
            # )


# Run the analysis
results = analyze_ben_training("ben_training_metrics_mean_chain-11.json")
print_analysis_results(results)
