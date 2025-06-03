import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

input_folder = "symbols"
csv_file = "features.csv"
profiles_folder = "profiles"
os.makedirs(profiles_folder, exist_ok=True)


def compute_features(image_path):
    img = Image.open(image_path).convert("L")
    img_np = np.array(img)
    binary = img_np < 128  # True - черный пиксель

    h, w = binary.shape
    area = h * w

    # Масса каждой четверти
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

    # Нормированные моменты
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


# Заголовки для CSV
header = [
    "char", "q1", "q2", "q3", "q4",
    "uq1", "uq2", "uq3", "uq4",
    "x_cg", "y_cg", "x_cg_norm", "y_cg_norm",
    "Ix", "Iy", "Ix_norm", "Iy_norm"
]

# Обработка изображений и сохранение
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(header)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            char = os.path.splitext(filename)[0]
            features = compute_features(os.path.join(input_folder, filename))

            # Сохраняем профили
            profile_x, profile_y = features[-2], features[-1]

            # Профиль X
            plt.figure(figsize=(4, 2))
            plt.bar(range(len(profile_x)), profile_x)
            plt.xticks(range(len(profile_x)))
            plt.title(f"{char} - X профиль")
            plt.tight_layout()
            plt.savefig(os.path.join(profiles_folder, f"{char}_x.png"))
            plt.close()

            # Профиль Y
            plt.figure(figsize=(2, 4))
            plt.barh(range(len(profile_y)), profile_y)
            plt.yticks(range(len(profile_y)))
            plt.title(f"{char} - Y профиль")
            plt.tight_layout()
            plt.savefig(os.path.join(profiles_folder, f"{char}_y.png"))
            plt.close()

            # Запись в CSV
            writer.writerow([char] + features[:-2])
