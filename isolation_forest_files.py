import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# === Choose which files ===
train_file = "syringe_all_drySend_syringe_1_features.csv"   # baseline / normal
test_file  = "syringe_syringe_1_drySend_syringe_1_features.csv"   # new / unknown

# === Load data ===
df_train = pd.read_csv(train_file)
df_test  = pd.read_csv(test_file)

feature_cols = [
    "Mean_Amplitude",
    "Std_Amplitude",
    "Skew_Val",
    "Spectral_Entropy",
    "Wavelet_Total_Energy",
    "Wavelet_Weighted_Scale",
    "STFT_Total_Energy",
    "STFT_Weighted_Frequency"
]

X_train = df_train[feature_cols].values
X_test  = df_test[feature_cols].values

# === Train Isolation Forest on baseline ===
clf = IsolationForest(contamination=0.02, random_state=42)
clf.fit(X_train)

# === Predict on test file ===
preds = clf.predict(X_test)              # 1 = normal, -1 = outlier
scores = clf.decision_function(X_test)   # higher = more normal

# === Add predictions and scores to dataframe ===
df_test["OutlierFlag"] = preds
df_test["AnomalyScore"] = scores

# === Display test results ===
print("\n=== Test Predictions ===")
print(df_test.head())
outlier_count = (preds == -1).sum()
print(f"\nOutlier count in test file: {outlier_count} / {len(preds)} ({100*outlier_count/len(preds):.2f}%)")

# === Optional: plot anomaly score distribution ===
plt.figure(figsize=(8,4))
plt.hist(scores, bins=50, color='steelblue', alpha=0.7)
plt.title(f"Anomaly Score Distribution — Test File: {test_file}")
plt.xlabel("Anomaly Score (higher = more normal)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# === Permutation-based feature importance ===
print("\n=== Running permutation-based feature importance... ===")

baseline_scores = clf.decision_function(X_test)
importances = []
n_repeats = 10  # repeat shuffling for stability

for i, feature in enumerate(feature_cols):
    changes = []
    for _ in range(n_repeats):
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, i])
        permuted_scores = clf.decision_function(X_permuted)
        # How much do anomaly scores change when this feature is destroyed?
        diff = np.mean(np.abs(baseline_scores - permuted_scores))
        changes.append(diff)
    importances.append(np.mean(changes))

# === Build importance DataFrame ===
imp_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n=== Permutation-based Feature Importance ===")
print(imp_df)

# === Plot importance chart ===
plt.figure(figsize=(8,4))
plt.barh(imp_df["Feature"], imp_df["Importance"], color="teal")
plt.xlabel("Mean Change in Anomaly Score")
plt.title("Permutation-based Feature Importance (Isolation Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
