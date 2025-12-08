#!/usr/bin/env python3
"""
Calculate p-values for comparing two evaluation reports.
Compares Healthy Trees Ratio and Episode Length across episodes.

For file1: Extracts the first Per-Episode Details section
For file2: Extracts only the "Nearest Fire Greedy Policy Results" Per-Episode Details section
         (Random Policy section is excluded)

Usage:
    python p_value.py <file1_path> <file2_path> [policy1_name] [policy2_name]

Example:
    python p_value.py evaluation/ppo/run22/evaluation_report.txt evaluation/heuristic/evaluation_report_single_fire.txt
"""

import re
import statistics
import sys
import os
from scipy import stats

def extract_data_from_report(file_path, policy_name, section_name=None):
    """
    Extract Healthy and Length data from evaluation report.

    Args:
        file_path: Path to evaluation report
        policy_name: Name of the policy
        section_name: Optional specific section to parse (e.g., "Nearest Fire Greedy Policy Results")
                      If None, parse the first "Per-Episode Details" section found

    Returns:
        Dictionary with 'healthy' and 'length' lists
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # If section_name is specified, find that section first
    search_content = content
    if section_name:
        section_start = content.find(section_name)
        if section_start == -1:
            raise ValueError(f"Could not find '{section_name}' section in {file_path}")

        # Find the next major section boundary
        next_results = content.find("Results", section_start + len(section_name))
        if next_results == -1:
            next_results = len(content)

        search_content = content[section_start:next_results]

    # Find the Per-Episode Details section
    per_episode_start = search_content.find("Per-Episode Details:")
    if per_episode_start == -1:
        raise ValueError(f"Could not find 'Per-Episode Details' section in {file_path}")

    healthy_values = []
    length_values = []

    episode_pattern = r"Episode \d+ \(seed=\d+\):\s+Healthy: ([\d.]+)%\s+Burnt:\s+[\d.]+%\s+Length:\s+(\d+) steps"
    matches = re.finditer(episode_pattern, search_content[per_episode_start:])

    for match in matches:
        healthy = float(match.group(1))
        length = int(match.group(2))
        healthy_values.append(healthy)
        length_values.append(length)

    if len(healthy_values) == 0 or len(length_values) == 0:
        raise ValueError(f"Could not extract any episodes from {file_path}")

    return {
        'healthy': healthy_values,
        'length': length_values,
        'policy': policy_name,
        'episodes': len(healthy_values)
    }

def calculate_p_values(file1_path, file2_path, policy1_name="Policy 1", policy2_name="Policy 2",
                       section1_name=None, section2_name=None):
    """
    Calculate p-values comparing two evaluation reports.
    Uses independent samples t-test (two-tailed).

    Args:
        file1_path: Path to first evaluation report
        file2_path: Path to second evaluation report
        policy1_name: Name of first policy
        policy2_name: Name of second policy
        section1_name: Optional section name for first file
        section2_name: Optional section name for second file (defaults to "Nearest Fire Greedy Policy Results")
    """

    # Parse both reports
    data1 = extract_data_from_report(file1_path, policy1_name, section1_name)
    # For file2, automatically use "Nearest Fire Greedy Policy Results" section if section2_name not specified
    if section2_name is None:
        section2_name = "Nearest Fire Greedy Policy Results"
    data2 = extract_data_from_report(file2_path, policy2_name, section2_name)

    # Perform independent samples t-tests
    healthy_t_stat, healthy_p_value = stats.ttest_ind(data1['healthy'], data2['healthy'])
    length_t_stat, length_p_value = stats.ttest_ind(data1['length'], data2['length'])

    # Calculate statistics
    mean1_healthy = sum(data1['healthy']) / len(data1['healthy'])
    mean2_healthy = sum(data2['healthy']) / len(data2['healthy'])
    std1_healthy = statistics.stdev(data1['healthy']) if len(data1['healthy']) > 1 else 0
    std2_healthy = statistics.stdev(data2['healthy']) if len(data2['healthy']) > 1 else 0

    mean1_length = sum(data1['length']) / len(data1['length'])
    mean2_length = sum(data2['length']) / len(data2['length'])
    std1_length = statistics.stdev(data1['length']) if len(data1['length']) > 1 else 0
    std2_length = statistics.stdev(data2['length']) if len(data2['length']) > 1 else 0

    # Print results
    print("=" * 80)
    print(f"Statistical Comparison: {data1['policy']} vs {data2['policy']}")
    print("=" * 80)
    print()

    print("Healthy Trees Ratio (%)")
    print("-" * 80)
    print(f"{data1['policy']}:")
    print(f"  Mean: {mean1_healthy:.2f}%")
    print(f"  Std:  {std1_healthy:.2f}%")
    print()
    print(f"{data2['policy']}:")
    print(f"  Mean: {mean2_healthy:.2f}%")
    print(f"  Std:  {std2_healthy:.2f}%")
    print()
    print(f"t-statistic: {healthy_t_stat:.4f}")
    print(f"p-value:     {healthy_p_value:.6f}")
    if healthy_p_value < 0.05:
        print("Result:      SIGNIFICANT difference (p < 0.05)")
    else:
        print("Result:      NO significant difference (p >= 0.05)")
    print()
    print()

    print("Episode Length (steps)")
    print("-" * 80)
    print(f"{data1['policy']}:")
    print(f"  Mean: {mean1_length:.2f} steps")
    print(f"  Std:  {std1_length:.2f} steps")
    print()
    print(f"{data2['policy']}:")
    print(f"  Mean: {mean2_length:.2f} steps")
    print(f"  Std:  {std2_length:.2f} steps")
    print()
    print(f"t-statistic: {length_t_stat:.4f}")
    print(f"p-value:     {length_p_value:.6f}")
    if length_p_value < 0.05:
        print("Result:      SIGNIFICANT difference (p < 0.05)")
    else:
        print("Result:      NO significant difference (p >= 0.05)")
    print()
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Healthy Trees Ratio p-value: {healthy_p_value:.6f}")
    print(f"Episode Length p-value:      {length_p_value:.6f}")
    print()

    return {
        'healthy_p_value': healthy_p_value,
        'length_p_value': length_p_value
    }

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python p_value.py <file1_path> <file2_path> [policy1_name] [policy2_name] [--section1 NAME] [--section2 NAME]")
        print()
        print("Examples:")
        print("  python p_value.py evaluation/ppo/run22/evaluation_report.txt evaluation/heuristic/evaluation_report_single_fire.txt")
        print("  python p_value.py eval1.txt eval2.txt PPO Heuristic")
        print("  python p_value.py eval1.txt eval2.txt PPO Heuristic --section1 'My Section' --section2 'Other Section'")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    policy1 = sys.argv[3] if len(sys.argv) > 3 else "Policy 1"
    policy2 = sys.argv[4] if len(sys.argv) > 4 else "Policy 2"

    section1 = None
    section2 = None

    # Parse optional section arguments
    if "--section1" in sys.argv:
        idx = sys.argv.index("--section1")
        if idx + 1 < len(sys.argv):
            section1 = sys.argv[idx + 1]

    if "--section2" in sys.argv:
        idx = sys.argv.index("--section2")
        if idx + 1 < len(sys.argv):
            section2 = sys.argv[idx + 1]

    # Check if files exist
    if not os.path.exists(file1):
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    if not os.path.exists(file2):
        print(f"Error: File not found: {file2}")
        sys.exit(1)

    try:
        results = calculate_p_values(file1, file2, policy1, policy2, section1, section2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
