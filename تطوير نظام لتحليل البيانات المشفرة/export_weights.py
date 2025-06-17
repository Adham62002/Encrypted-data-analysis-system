import numpy as np
import joblib

LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
weights = []
biases = []

for label in LABELS:
    try:
        model = joblib.load(f"models/{label}_classifier.pkl")
        weights.append(model.coef_[0])   # ← [2048]
        biases.append(model.intercept_[0])
    except Exception as e:
        print(f"⚠️ فشل في تحميل {label}: {e}")
        weights.append(np.zeros(2048))
        biases.append(0.0)

weights = np.array(weights)  # [8, 2048]
biases = np.array(biases)    # [8]

np.save("models/weights.npy", weights)
np.save("models/bias.npy", biases)

print("✅ تم حفظ weights.npy و bias.npy بنجاح")
