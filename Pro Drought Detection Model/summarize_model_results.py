# summarize_model_results.py

import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Use Anti-Grain Geometry backend (no GUI)

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("models", exist_ok=True)

# Replace these with actual evaluation results
results = {
    "Model": ["A1", "A2", "A4", "A6"],
    "Weighted F1": [0.9675, 0.9286, 0.9459, 0.8942],
    "Precision": [0.9762, 0.9286, 0.9667, 0.8622],
    "Recall": [0.9643, 0.89286, 0.9375, 0.9286],
    "ROC-AUC": [0.9952, 0.9784, 0.9916, 0.6971]
}

# Create DataFrame
df = pd.DataFrame(results)

# Save CSV summary
csv_path = "models/model_performance_comparison_2.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Saved CSV summary to {csv_path}")

# Plot chart: Weighted F1 and ROC-AUC
plt.figure(figsize=(10, 6))
df.set_index("Model")[["Weighted F1", "ROC-AUC"]].plot(kind="bar")
plt.title("Model Comparison: F1-Score & ROC-AUC")
plt.ylabel("Score")
plt.ylim(0.7, 1.0)
plt.grid(axis='y')
plt.tight_layout()

# Save chart
chart_path = "models/model_performance_comparison.png"
plt.savefig(chart_path)
print(f"✅ Saved comparison chart to {chart_path}")
plt.show()
