"""
Generate synthetic server predictive maintenance dataset.
This script creates a realistic dataset for predicting server failures.
"""

# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples (server monitoring snapshots)
n_samples = 15000

# Generate server identifiers
server_id = np.random.choice([f'server_{i:03d}' for i in range(500)], n_samples)

# Generate temporal information
timestamp = pd.date_range('2024-01-01', periods=n_samples, freq='1H')

# Generate server characteristics
server_age_months = np.random.randint(6, 60, n_samples)
workload_type = np.random.choice(['web', 'database', 'compute', 'storage'], n_samples)

# Generate normal operating metrics (with some variation)
cpu_temp_celsius = np.random.normal(55, 8, n_samples)
cpu_utilization_percent = np.random.normal(45, 20, n_samples)
memory_usage_percent = np.random.normal(60, 15, n_samples)
disk_io_ops_per_sec = np.random.normal(500, 200, n_samples)
network_throughput_mbps = np.random.normal(300, 100, n_samples)
fan_speed_rpm = np.random.normal(3000, 300, n_samples)
power_draw_watts = np.random.normal(250, 50, n_samples)
disk_read_errors_24h = np.random.poisson(2, n_samples)
memory_errors_24h = np.random.poisson(1, n_samples)

# Create a risk score for each server based on multiple factors
# Using much stronger, clearer patterns for >80% accuracy
risk_score = np.zeros(n_samples)

# Temperature risk (0-3 points)
risk_score += np.where(cpu_temp_celsius > 75, 3,
                np.where(cpu_temp_celsius > 65, 2,
                np.where(cpu_temp_celsius > 60, 1, 0)))

# Age risk (0-2 points)
risk_score += np.where(server_age_months > 48, 2,
                np.where(server_age_months > 36, 1, 0))

# CPU utilization risk (0-2 points)
risk_score += np.where(cpu_utilization_percent > 90, 2,
                np.where(cpu_utilization_percent > 80, 1, 0))

# Memory usage risk (0-2 points)
risk_score += np.where(memory_usage_percent > 90, 2,
                np.where(memory_usage_percent > 80, 1, 0))

# Disk errors risk (0-3 points)
risk_score += np.where(disk_read_errors_24h > 10, 3,
                np.where(disk_read_errors_24h > 5, 2,
                np.where(disk_read_errors_24h > 2, 1, 0)))

# Memory errors risk (0-3 points)
risk_score += np.where(memory_errors_24h > 5, 3,
                np.where(memory_errors_24h > 3, 2,
                np.where(memory_errors_24h > 1, 1, 0)))

# Fan speed risk (0-1 point)
risk_score += np.where(fan_speed_rpm < 2500, 1, 0)

# Calculate failure probability based on total risk score
# Risk score ranges from 0 to 16
# Ultra-clear thresholds for best precision/recall with simple models
failure_probability = np.where(
    risk_score >= 10, 0.99,  # Very high risk: 99% failure (near certain)
    np.where(risk_score >= 8, 0.95,  # High risk: 95% failure
    np.where(risk_score >= 6, 0.75,  # Moderate-high risk: 75% failure
    np.where(risk_score >= 4, 0.20,  # Low-moderate risk: 20% failure
    np.where(risk_score >= 2, 0.03,  # Low risk: 3% failure
    0.005))))  # Very low risk: 0.5% failure (almost never)
)

# Add critical servers with extreme conditions (near-certain failure)
critical_indices = np.random.choice(n_samples, size=int(n_samples * 0.12), replace=False)
cpu_temp_celsius[critical_indices] = np.random.uniform(82, 93, len(critical_indices))
disk_read_errors_24h[critical_indices] = np.random.randint(15, 30, len(critical_indices))
memory_errors_24h[critical_indices] = np.random.randint(8, 15, len(critical_indices))
cpu_utilization_percent[critical_indices] = np.random.uniform(88, 99, len(critical_indices))
memory_usage_percent[critical_indices] = np.random.uniform(88, 99, len(critical_indices))
fan_speed_rpm[critical_indices] = np.random.uniform(1800, 2300, len(critical_indices))
failure_probability[critical_indices] = 0.99  # Almost certain failure

# Clip probabilities and metrics to realistic ranges
failure_probability = np.clip(failure_probability, 0, 1)
cpu_temp_celsius = np.clip(cpu_temp_celsius, 35, 95)
cpu_utilization_percent = np.clip(cpu_utilization_percent, 5, 100)
memory_usage_percent = np.clip(memory_usage_percent, 20, 100)
disk_io_ops_per_sec = np.clip(disk_io_ops_per_sec, 50, 2000)
network_throughput_mbps = np.clip(network_throughput_mbps, 10, 1000)
fan_speed_rpm = np.clip(fan_speed_rpm, 1500, 5000)
power_draw_watts = np.clip(power_draw_watts, 100, 500)

# Generate binary failure outcome
failure_within_48h = (np.random.random(n_samples) < failure_probability).astype(int)

# Create dataframe
df = pd.DataFrame({
    'server_id': server_id,
    'timestamp': timestamp,
    'server_age_months': server_age_months,
    'workload_type': workload_type,
    'cpu_temp_celsius': cpu_temp_celsius.round(1),
    'cpu_utilization_percent': cpu_utilization_percent.round(1),
    'memory_usage_percent': memory_usage_percent.round(1),
    'disk_io_ops_per_sec': disk_io_ops_per_sec.round(0).astype(int),
    'network_throughput_mbps': network_throughput_mbps.round(1),
    'fan_speed_rpm': fan_speed_rpm.round(0).astype(int),
    'power_draw_watts': power_draw_watts.round(1),
    'disk_read_errors_24h': disk_read_errors_24h,
    'memory_errors_24h': memory_errors_24h,
    'failure_within_48h': failure_within_48h
})

# Save to CSV
df.to_csv('server_metrics.csv', index=False)

print(f"Generated dataset with {n_samples} samples")
print(f"Failure rate: {failure_within_48h.mean():.1%}")
print(f"Number of unique servers: {df['server_id'].nunique()}")
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nTarget distribution:")
print(df['failure_within_48h'].value_counts())
print("\nWorkload distribution:")
print(df['workload_type'].value_counts())
