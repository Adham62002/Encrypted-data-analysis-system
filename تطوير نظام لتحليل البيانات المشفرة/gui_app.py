import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import datetime

import tenseal as ts
import numpy as np
from extract_features import extract_feature
from encrypted_classifier import classify_encrypted

LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# إنشاء السياق CKKS
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

# تصنيف صورة واحدة
def classify_image_encrypted(img_path):
    features = extract_feature(img_path)
    context = create_context()
    enc_vector = ts.ckks_vector(context, features)
    prediction_vector = classify_encrypted(enc_vector)

    detected = []
    log_lines = []
    for i, score in enumerate(prediction_vector):
        line = f"{LABELS[i]} → {score:.3f}"
        log_lines.append(line)
        if score > 0:
            detected.append(LABELS[i])

    summary = " | ".join(detected) if detected else "✅ لا توجد أمراض"
    return summary, log_lines

# إنشاء واجهة المستخدم
class EyeDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Disease Classifier (Homomorphic Encryption)")
        self.root.geometry("700x600")
        self.image_path = None

        # واجهة
        self.label_title = tk.Label(root, text="🔬 تصنيف أمراض العين باستخدام التشفير المتجانس", font=("Arial", 14, "bold"))
        self.label_title.pack(pady=10)

        self.btn_choose = tk.Button(root, text="📂 اختر صورة", command=self.load_image)
        self.btn_choose.pack(pady=5)

        self.canvas = tk.Label(root)
        self.canvas.pack(pady=10)

        self.btn_predict = tk.Button(root, text="🔍 تشخيص مشفّر", command=self.predict)
        self.btn_predict.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), fg="green")
        self.result_label.pack(pady=10)

        self.btn_save = tk.Button(root, text="💾 حفظ التقرير", command=self.save_report, state=tk.DISABLED)
        self.btn_save.pack(pady=5)

        self.report_log = ""

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            img = ImageTk.PhotoImage(img)
            self.canvas.configure(image=img)
            self.canvas.image = img
            self.result_label.config(text="")
            self.report_log = ""
            self.btn_save.config(state=tk.DISABLED)

    def predict(self):
        if not self.image_path:
            messagebox.showerror("خطأ", "يرجى اختيار صورة أولًا.")
            return

        try:
            summary, lines = classify_image_encrypted(self.image_path)
            self.result_label.config(text=f"النتيجة: {summary}")
            log = f"📅 التاريخ: {datetime.datetime.now()}\n📁 الصورة: {os.path.basename(self.image_path)}\n\n"
            log += "\n".join(lines)
            log += f"\n\nالنتيجة النهائية: {summary}\n"
            self.report_log = log
            self.btn_save.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("خطأ أثناء المعالجة", str(e))

    def save_report(self):
        if not self.report_log:
            return
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(os.path.join(report_dir, filename), "w", encoding="utf-8") as f:
            f.write(self.report_log)
        messagebox.showinfo("تم الحفظ", f"تم حفظ التقرير في: {filename}")

# تشغيل التطبيق
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeDiseaseApp(root)
    root.mainloop()
