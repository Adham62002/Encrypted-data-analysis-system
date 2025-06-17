import pandas as pd
import numpy as np
import os
from extract_features import extract_feature

LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
df = pd.read_csv("data/labels.csv")

X = []
y = []

# 👇 حدد هنا عدد الصور الكلي (مثلاً: 300 صورة = 150 مريض × 2 عين)
MAX_SAMPLES = 300
counter = 0

for i, row in df.iterrows():
    for side in ["Left-Fundus", "Right-Fundus"]:
        if counter >= MAX_SAMPLES:
            break

        img_name = row[side]
        img_path = os.path.join("data/Full_Training_Data", img_name)

        if not os.path.exists(img_path):
            print(f"❌ الصورة غير موجودة: {img_path}")
            continue

        try:
            print(f"[{counter+1}/{MAX_SAMPLES}] 🖼️ استخراج ميزات من: {img_name}")
            features = extract_feature(img_path)
            X.append(features)

            label_vector = [row[label] for label in LABELS]
            y.append(label_vector)

            counter += 1
        except Exception as e:
            print(f"⚠️ خطأ في {img_name}: {e}")

# التحويل إلى مصفوفات وحفظها
X = np.array(X)
y = np.array(y)

os.makedirs("features", exist_ok=True)
np.save("features/X.npy", X)
np.save("features/y.npy", y)

print(f"✅ تم حفظ {counter} ميزة في features/X.npy و features/y.npy")
