import numpy as np
from scipy.stats import ks_2samp

X_old = np.random.normal(50, 10, 1000)

# new data, shifted
X_new = np.random.normal(70, 10, 1000)

stat, p_value = ks_2samp(X_old, X_new)

print(f"KS statistic: {stat}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("Drift detected")
else:
    print("No data drift detected")

# KS test compares distributions
# No ML knowledge required
# Used widely in production monitoring
# This is what tools like Evidently do internally