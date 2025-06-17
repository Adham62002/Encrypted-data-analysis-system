import tenseal as ts
import numpy as np

# تحميل الأوزان والانحيازات
W = np.load("models/weights.npy")  # ← [8, 2048]
b = np.load("models/bias.npy")     # ← [8]

# التصنيف المشفر: لكل مرض نحسب dot و bias، ثم نفك التشفير
def classify_encrypted(enc_vector):
    results = []
    for i in range(len(b)):
        # تطبيق Wx + b على البيانات المشفرة
        res = enc_vector.dot(W[i]) + b[i]
        decrypted = res.decrypt()[0]  # فك التشفير لنتيجة واحدة (float)
        results.append(decrypted)
    return results  # ← قائمة من القيم الحقيقية
