# Requires: pip install numpy
import numpy as np

class PageHinkley:
    def __init__(self, delta=0.005, threshold=50.0, min_instances=30):
        self.delta = delta
        self.threshold = threshold
        self.min_instances = min_instances
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.mT = 0.0
        self.n = 0

    def add(self, x):
        self.n += 1
        prev_mean = self.mean
        self.mean += (x - self.mean) / self.n
        self.mT = min(self.mT, (self.mT + x - self.mean - self.delta))
        # compute cumulative deviation
        diff = (x - self.mean - self.delta)
        self.mT = min(self.mT, diff if self.n==1 else self.mT + diff)
        if self.n < self.min_instances:
            return False
        # signal if cumulative deviation minus minimum exceeds threshold
        if ( (self.mean - self.mT) > self.threshold ):
            return True
        return False

def cusum_detector(stream, k=0.5, h=8.0):
    s_pos = 0.0
    s_neg = 0.0
    for i, x in enumerate(stream):
        s_pos = max(0.0, s_pos + x - k)
        s_neg = min(0.0, s_neg + x + k)
        if s_pos > h or abs(s_neg) > h:
            return i, x
    return None, None

def psi(expected, actual, buckets=10):
    bins = np.linspace(min(expected.min(), actual.min()),
                       max(expected.max(), actual.max()), buckets+1)
    exp_pct, _ = np.histogram(expected, bins=bins)
    act_pct, _ = np.histogram(actual, bins=bins)
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()
    # avoid zeros
    act_pct = np.where(act_pct==0, 1e-6, act_pct)
    exp_pct = np.where(exp_pct==0, 1e-6, exp_pct)
    return np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct))

# Example usage
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    stable = rng.normal(0,1,500)
    drift = rng.normal(1.5,1,200)
    stream = np.concatenate([stable, drift])

    ph = PageHinkley(delta=0.01, threshold=30.0, min_instances=50)
    for i, x in enumerate(stream):
        if ph.add(x):
            print("Page-Hinkley detected change at", i, "value", round(x,3))
            break

    idx, val = cusum_detector(stream, k=0.5, h=8.0)
    if idx is not None:
        print("CUSUM detected change at", idx, "value", round(val,3))

    ref = rng.normal(0,1,1000)
    new = rng.normal(0.3,1,1000)
    print("PSI:", round(psi(ref,new, buckets=10),4))


# it implements a streaming Page‑Hinkley detector, a simple CUSUM detector, and a 
# batch PSI function (all runnable with only numpy). Run locally in Ottawa now; the 
# code is copy‑paste runnable and includes example streams.

# METHOD       | TYPE   | STRENGTH                            | WHEN TO USE
# Page‑Hinkley | Online | Simple, robust to noise             | Detect mean shifts in streaming numeric features.
# CUSUM        | Online | Fast for small persistent shifts    | Low-latency detection of gradual changes.
# PSI          | Batch  | Interpretable, regulatory-friendly  | Periodic checks between reference and new windows.

# Practical tips & next steps
# - Tuning: delta/min_instances (Page‑Hinkley) and k/h (CUSUM) control sensitivity — increase to reduce false positives.
# - Integration: log alerts/PSI to MLflow as metrics; store reference windows with DVC for reproducibility.
# - Scale: wrap the detector in a FastAPI worker or run inside a Dask task for parallel streams.
# - Further reading / implementations: pure‑Python ADWIN candidates exist on GitHub if you later want ADWIN behavior.
