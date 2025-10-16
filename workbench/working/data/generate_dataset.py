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

# Calculate failure probability based on risk factors
# Higher risk: high temps, old servers, high utilization, errors
failure_probability = (
    0.01 +  # baseline failure rate
    0.001 * np.maximum(0, cpu_temp_celsius - 60) +  # temp above 60°C
    0.0005 * server_age_months +  # older servers
    0.0003 * np.maximum(0, cpu_utilization_percent - 80) +  # high CPU
    0.0002 * np.maximum(0, memory_usage_percent - 80) +  # high memory
    0.002 * disk_read_errors_24h +  # disk errors
    0.003 * memory_errors_24h +  # memory errors
    0.0001 * np.maximum(0, 65 - fan_speed_rpm / 50)  # low fan speed
)

# Add some servers with critical conditions (imminent failure)
critical_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
cpu_temp_celsius[critical_indices] += np.random.uniform(15, 25, len(critical_indices))
disk_read_errors_24h[critical_indices] += np.random.poisson(10, len(critical_indices))
memory_errors_24h[critical_indices] += np.random.poisson(5, len(critical_indices))
failure_probability[critical_indices] += 0.4

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
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df['failure_within_48h'].value_counts())
print(f"\nWorkload distribution:")
print(df['workload_type'].value_counts())
