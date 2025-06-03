import cv2
import numpy as np
import matplotlib.pyplot as plt

# Параметры GLCM
distances = [1]
angles_deg = [45, 135, 225, 315]
angles_rad = [np.deg2rad(a) for a in angles_deg]

def load_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def equalize_image(img):
    return cv2.equalizeHist(img)

def compute_glcm(img, distance, angle):
    max_gray = int(img.max())
    levels = max_gray + 1
    glcm = np.zeros((levels, levels), dtype=np.uint32)
    rows, cols = img.shape
    dx = int(round(np.cos(angle) * distance))
    dy = int(round(np.sin(angle) * distance))

    for i in range(rows):
        for j in range(cols):
            row2 = i + dy
            col2 = j + dx
            if 0 <= row2 < rows and 0 <= col2 < cols:
                glcm[img[i, j], img[row2, col2]] += 1
    return glcm

def log_normalize(mat):
    # Логарифмическое нормирование для улучшения видимости
    return np.log1p(mat) / np.log1p(mat).max()

def compute_av_d(glcm_matrices):
    # AV — среднее значение всех элементов GLCM
    # D — стандартное отклонение всех элементов GLCM
    all_values = np.concatenate([g.flatten() for g in glcm_matrices])
    av = np.mean(all_values)
    d = np.std(all_values)
    return av, d

# Новая функция для числового анализа по каждому углу
def compute_av_d_per_glcm(glcm_matrices, angles_deg):
    print("Числовой анализ признаков AV и D по углам:")
    for i, angle in enumerate(angles_deg):
        av = np.mean(glcm_matrices[i])
        d = np.std(glcm_matrices[i])
        print(f"Угол {angle}°: AV = {av:.6f}, D = {d:.6f}")

def plot_histograms(original, equalized, ax_orig, ax_eq):
    ax_orig.hist(original.ravel(), bins=256, color='black')
    ax_orig.set_title('Гистограмма яркости исходного')
    ax_eq.hist(equalized.ravel(), bins=256, color='black')
    ax_eq.set_title('Гистограмма яркости выравненного')

def process_and_plot(path):
    original = load_image_gray(path)
    equalized = equalize_image(original)

    glcm_orig = [compute_glcm(original, distances[0], a) for a in angles_rad]
    glcm_eq = [compute_glcm(equalized, distances[0], a) for a in angles_rad]

    # Вот здесь добавляем вызов новой функции для числового анализа
    print(f"\nАнализ для изображения: {path}")
    compute_av_d_per_glcm(glcm_orig, angles_deg)
    compute_av_d_per_glcm(glcm_eq, angles_deg)

    av_orig, d_orig = compute_av_d(glcm_orig)
    av_eq, d_eq = compute_av_d(glcm_eq)

    fig, axs = plt.subplots(5, 4, figsize=(15, 15))
    fig.suptitle(f'Результаты для {path}', fontsize=16)

    # Показываем исходное и выравненное изображение
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title('Исходное изображение')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(equalized, cmap='gray')
    axs[0, 1].set_title('Контрастированное (выравненное)')
    axs[0, 1].axis('off')

    # Гистограммы
    plot_histograms(original, equalized, axs[0, 2], axs[0, 3])

    # Визуализация GLCM (логарифмически нормированные)
    for i, angle in enumerate(angles_deg):
        axs[1, i].imshow(log_normalize(glcm_orig[i]), cmap='gray')
        axs[1, i].set_title(f'GLCM исходное\nугол {angle}°')
        axs[1, i].axis('off')

        axs[2, i].imshow(log_normalize(glcm_eq[i]), cmap='gray')
        axs[2, i].set_title(f'GLCM контрастированное\nугол {angle}°')
        axs[2, i].axis('off')

    # Вывод признаков AV и D (исходное и контрастированное)
    axs[3, 0].text(0, 0.5, f'AV исходное: {av_orig:.6f}\nD исходное: {d_orig:.6f}', fontsize=12)
    axs[3, 0].axis('off')
    axs[3, 1].text(0, 0.5, f'AV контрастированное: {av_eq:.6f}\nD контрастированное: {d_eq:.6f}', fontsize=12)
    axs[3, 1].axis('off')

    # Пустые места, чтобы не мешать
    for j in range(2, 4):
        axs[3, j].axis('off')
    for i in range(4, 5):
        for j in range(4):
            axs[i, j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Пути к изображениям — поменяй на свои
paths = [
    "Photos/ph1.png",
    "Photos/ph2.png",
    "Photos/ph5.png"
]

for p in paths:
    process_and_plot(p)
