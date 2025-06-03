import os
import csv
import numpy as np
from PIL import Image
from mass import compute_features

# Пути к файлам и папкам
template_csv = "../features.csv"
input_dir = "../Photo/phrase"
output_file = "../hypotheses.txt"

# Чтение эталонных признаков из CSV
templates = {}
with open(template_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        char = row["char"]
        features = [
            float(row["x_cg_norm"]),
            float(row["y_cg_norm"]),
            float(row["Ix_norm"]),
            float(row["Iy_norm"]),
            float(row["uq1"]) + float(row["uq2"]) + float(row["uq3"]) + float(row["uq4"])
        ]
        templates[char] = np.array(features)

# Обработка файлов из папки symbols
results = []
symbols = []

image_files = sorted(os.listdir(input_dir))
for filename in image_files:
    if filename.lower().endswith((".png", ".bmp")):
        symbol = os.path.splitext(filename)[0]  # имя файла без расширения
        symbols.append(symbol)

        img_path = os.path.join(input_dir, filename)
        features = compute_features(img_path)

        # Вытаскиваем нужные признаки
        x_cg_norm, y_cg_norm = features[10], features[11]
        Ix_norm, Iy_norm = features[14], features[15]
        mass = features[4] + features[5] + features[6] + features[7]

        test_vector = np.array([x_cg_norm, y_cg_norm, Ix_norm, Iy_norm, mass])

        # Вычисляем расстояния и похожесть с эталонами
        distances = []
        for char, tpl_vector in templates.items():
            dist = np.linalg.norm(test_vector - tpl_vector)
            similarity = 1 / (1 + dist)
            distances.append((char, round(similarity, 4)))

        distances.sort(key=lambda x: x[1], reverse=True)
        results.append(distances)

# Сохраняем гипотезы в файл и выводим на экран
with open(output_file, "w", encoding="utf-8") as f:
    for symbol, hypotheses in zip(symbols, results):
        top_hypotheses = ", ".join(f"{char} ({score})" for char, score in hypotheses[:5])
        line = f"{symbol}: {top_hypotheses}"
        print(line)
        f.write(line + "\n")