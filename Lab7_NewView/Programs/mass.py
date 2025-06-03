import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Папки и пути
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "../Photo/alphabet")
csv_file = os.path.join(script_dir, "../features.csv")
profiles_folder = os.path.join(script_dir, "../Photo/alphabet_phofs")
profiles_csv_folder = os.path.join(profiles_folder, "csv")

# Создание папок, если их нет
os.makedirs(profiles_folder, exist_ok=True)
os.makedirs(profiles_csv_folder, exist_ok=True)

# Функция вычисления признаков
def compute_features(image_path):
    img = Image.open(image_path).convert("L")
    img_np = np.array(img)
    binary = img_np < 128  # True - чёрный пиксель

    if not np.any(binary):
        print(f"⚠️  Нет чёрных пикселей в изображении: {image_path}")
        return None

    h, w = binary.shape
    area = h * w

    # Масса четвертей
    q1 = binary[0:h // 2, 0:w // 2].sum()
    q2 = binary[0:h // 2, w // 2:].sum()
    q3 = binary[h // 2:, 0:w // 2].sum()
    q4 = binary[h // 2:, w // 2:].sum()

    # Удельные массы
    quarter_area = (h // 2) * (w // 2)
    uq1 = q1 / quarter_area
    uq2 = q2 / quarter_area
    uq3 = q3 / quarter_area
    uq4 = q4 / quarter_area

    # Центр тяжести
    coords = np.argwhere(binary)
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    x_cg = x_coords.mean()
    y_cg = y_coords.mean()

    # Нормированный центр тяжести
    x_cg_norm = x_cg / w
    y_cg_norm = y_cg / h

    # Моменты инерции
    Ix = ((x_coords - x_cg) ** 2).sum()
    Iy = ((y_coords - y_cg) ** 2).sum()
    Ix_norm = Ix / (w ** 2 * binary.sum())
    Iy_norm = Iy / (h ** 2 * binary.sum())

    # Профили
    profile_x = binary.sum(axis=0)
    profile_y = binary.sum(axis=1)

    return [
        q1, q2, q3, q4,
        uq1, uq2, uq3, uq4,
        x_cg, y_cg,
        x_cg_norm, y_cg_norm,
        Ix, Iy,
        Ix_norm, Iy_norm,
        profile_x, profile_y
    ]

# Заголовки
header = [
    "char", "q1", "q2", "q3", "q4",
    "uq1", "uq2", "uq3", "uq4",
    "x_cg", "y_cg", "x_cg_norm", "y_cg_norm",
    "Ix", "Iy", "Ix_norm", "Iy_norm"
]

# Сбор признаков и запись CSV
processed = 0
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(header)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
            char = os.path.splitext(filename)[0]
            path = os.path.join(input_folder, filename)
            print(f"🔄 Обработка: {filename}")
            features = compute_features(path)
            if features is None:
                continue

            # Получение профилей
            profile_x, profile_y = features[-2], features[-1]

            # Сохранение графиков профилей
            plt.figure(figsize=(4, 2))
            plt.bar(range(len(profile_x)), profile_x)
            plt.title(f"{char} - X профиль")
            plt.tight_layout()
            plt.savefig(os.path.join(profiles_folder, f"{char}_x.png"))
            plt.close()

            plt.figure(figsize=(2, 4))
            plt.barh(range(len(profile_y)), profile_y)
            plt.title(f"{char} - Y профиль")
            plt.tight_layout()
            plt.savefig(os.path.join(profiles_folder, f"{char}_y.png"))
            plt.close()

            # Сохранение самих профилей в CSV
            with open(os.path.join(profiles_csv_folder, f"{char}_x.csv"), "w", newline="") as xf:
                writer_x = csv.writer(xf)
                writer_x.writerow(["x", "value"])
                for i, val in enumerate(profile_x):
                    writer_x.writerow([i, val])

            with open(os.path.join(profiles_csv_folder, f"{char}_y.csv"), "w", newline="") as yf:
                writer_y = csv.writer(yf)
                writer_y.writerow(["y", "value"])
                for i, val in enumerate(profile_y):
                    writer_y.writerow([i, val])

            # Запись в общий CSV
            writer.writerow([char] + features[:-2])
            processed += 1

print(f"\n✅ Обработка завершена. Всего обработано файлов: {processed}")
