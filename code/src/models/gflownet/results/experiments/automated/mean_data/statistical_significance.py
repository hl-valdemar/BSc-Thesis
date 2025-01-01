import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load the JSON data and convert to pandas DataFrame."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data["metrics_history"])


def split_into_stages(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into early, mid, and late stages."""
    total_steps = len(df)
    early_end = int(total_steps * 0.33)
    mid_end = int(total_steps * 0.66)

    early = df.iloc[:early_end]
    mid = df.iloc[early_end:mid_end]
    late = df.iloc[mid_end:]

    return early, mid, late


def calculate_robust_effect_size(stage1: pd.Series, stage2: pd.Series) -> float:
    """Calculate effect size with robustness to near-zero variances."""
    diff = stage2.mean() - stage1.mean()
    # Use pooled standard deviation with regularization
    var1, var2 = stage1.var(), stage2.var()
    pooled_std = np.sqrt(
        (var1 + var2) / 2 + 1e-10
    )  # Add small constant for numerical stability

    return diff / pooled_std if pooled_std > 0 else np.inf * np.sign(diff)


def perform_statistical_tests(
    early: pd.DataFrame, mid: pd.DataFrame, late: pd.DataFrame, metrics: List[str]
) -> Dict:
    """Perform statistical tests between stages for specified metrics."""
    results = {}
    stages = [
        ("early-mid", early, mid),
        ("mid-late", mid, late),
        ("early-late", early, late),
    ]

    for metric in metrics:
        metric_results = {}
        for stage_name, stage1, stage2 in stages:
            # For nearly identical data, use non-parametric test by default
            if stage1[metric].std() < 1e-6 or stage2[metric].std() < 1e-6:
                stat, p_val = stats.mannwhitneyu(
                    stage1[metric], stage2[metric], alternative="two-sided"
                )
                test_name = "Mann-Whitney U (due to uniform data)"
            else:
                # Perform Shapiro-Wilk test for normality (more robust for smaller samples)
                _, p_val_1 = stats.shapiro(stage1[metric])
                _, p_val_2 = stats.shapiro(stage2[metric])

                if p_val_1 > 0.05 and p_val_2 > 0.05:
                    stat, p_val = stats.ttest_ind(stage1[metric], stage2[metric])
                    test_name = "t-test"
                else:
                    stat, p_val = stats.mannwhitneyu(
                        stage1[metric], stage2[metric], alternative="two-sided"
                    )
                    test_name = "Mann-Whitney U"

            # Calculate robust effect size
            effect_size = calculate_robust_effect_size(stage1[metric], stage2[metric])

            metric_results[stage_name] = {
                "test_used": test_name,
                "statistic": float(stat),  # Convert numpy types to native Python
                "p_value": float(p_val),
                "significant": p_val < 0.05,
                "mean_diff": float(stage2[metric].mean() - stage1[metric].mean()),
                "median_diff": float(stage2[metric].median() - stage1[metric].median()),
                "effect_size": float(effect_size),
            }

        results[metric] = metric_results

    return results


def analyze_training_data(file_path: str, metrics: List[str]) -> Dict:
    """Main analysis function."""
    # Load and prepare data
    df = load_and_prepare_data(file_path)

    # Split into stages
    early, mid, late = split_into_stages(df)

    # Perform statistical tests
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        results = perform_statistical_tests(early, mid, late, metrics)

    # Calculate summary statistics
    summary_stats = {}
    for metric in metrics:
        summary_stats[metric] = {
            "early": {
                "mean": float(early[metric].mean()),
                "std": float(early[metric].std()),
                "median": float(early[metric].median()),
            },
            "mid": {
                "mean": float(mid[metric].mean()),
                "std": float(mid[metric].std()),
                "median": float(mid[metric].median()),
            },
            "late": {
                "mean": float(late[metric].mean()),
                "std": float(late[metric].std()),
                "median": float(late[metric].median()),
            },
        }

    return {"statistical_tests": results, "summary_statistics": summary_stats}


def print_analysis_results(results: Dict, metrics: List[str]):
    """Print analysis results in a clear, formatted way."""
    for metric in metrics:
        print(f"\n{'='*50}")
        print(f"Analysis for {metric}")
        print(f"{'='*50}")

        # Print summary statistics
        print("\nSummary Statistics:")
        for stage in ["early", "mid", "late"]:
            stats = results["summary_statistics"][metric][stage]
            print(f"\n{stage.capitalize()} stage:")
            print(f"  Mean   : {stats['mean']:.6g}")
            print(f"  Std    : {stats['std']:.6g}")
            print(f"  Median : {stats['median']:.6g}")

        # Print statistical test results
        print("\nStatistical Tests:")
        for comparison, test_results in results["statistical_tests"][metric].items():
            print(f"\n{comparison.upper()}:")
            print(f"  Test used    : {test_results['test_used']}")
            print(f"  P-value      : {test_results['p_value']:.6g}")
            print(f"  Significant  : {'Yes' if test_results['significant'] else 'No'}")
            print(f"  Effect size  : {test_results['effect_size']:.6g}")
            print(f"  Mean diff    : {test_results['mean_diff']:.6g}")
            print(f"  Median diff  : {test_results['median_diff']:.6g}")


# Example usage:
metrics_to_analyze = [
    "trajectory_balance_loss",
    "terminal_reward",
    "exploration_ratio",
    "forward_entropy",
    "backward_entropy",
]

results = analyze_training_data(
    "gflownet_training_metrics_mean_chain-11.json", metrics_to_analyze
)
print_analysis_results(results, metrics_to_analyze)
