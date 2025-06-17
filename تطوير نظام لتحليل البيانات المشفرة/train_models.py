from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import os

X = np.load("features/X.npy")
y = np.load("features/y.npy")

os.makedirs("models", exist_ok=True)

LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

for i, label in enumerate(LABELS):
    unique_classes = np.unique(y[:, i])
    if len(unique_classes) < 2:
        print(f"⚠️ تخطينا {label} لأنه لا يحتوي على صنفين (البيانات: {unique_classes})")
        continue

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y[:, i])
    joblib.dump(model, f"models/{label}_classifier.pkl")
    print(f"✅ تم تدريب وحفظ نموذج {label}")
