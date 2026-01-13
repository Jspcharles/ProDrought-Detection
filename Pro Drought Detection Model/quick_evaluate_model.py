import pandas as pd
import matplotlib.pyplot as plt

# Manually entered results (replace with your actual values)
results = {
    "Model": ["A1", "A2", "A4", "A6"],
    "Weighted F1": [0.9675, 0.9286, 0.9459, 0.8942],
    "Precision": [0.9762, 0.9286, 0.9667, 0.8622],
    "Recall": [0.9643, 0.89286, 0.9375, 0.9286],
    "ROC-AUC": [0.9952, 0.9784, 0.9916, 0.6971]
}

df = pd.DataFrame(results)

# Save to CSV
csv_path = "models/model_performance_comparison.csv"
df.to_csv(csv_path, index=False)

# Bar chart
df.set_index("Model")[["Weighted F1", "ROC-AUC"]].plot(kind="bar", figsize=(10, 6))
plt.title("Model Comparison: F1-Score & ROC-AUC")
plt.ylabel("Score")
plt.ylim(0.7, 1.0)
plt.grid(axis='y')
plt.tight_layout()

plt_path = "models/model_performance_comparison.png"
plt.savefig(plt_path)
plt.show()
