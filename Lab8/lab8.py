import cv2
import numpy as np
import matplotlib.pyplot as plt

# Параметры GLCM
distances = [1]
angles_deg = [45, 135, 225, 315]
angles_rad = [np.deg2rad(a) for a in angles_deg]


def load_image_hsl(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # В OpenCV HSL называется HLS
    return img, hsl


def equalize_l_channel(hsl_img):
    l_channel = hsl_img[:, :, 1]
    eq_l = cv2.equalizeHist(l_channel)
    hsl_eq = hsl_img.copy()
    hsl_eq[:, :, 1] = eq_l
    return hsl_eq


def hsl_to_bgr(hsl_img):
    return cv2.cvtColor(hsl_img, cv2.COLOR_HLS2BGR)


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
    return np.log1p(mat) / np.log1p(mat).max()


def compute_av_d(glcm_matrices):
    all_values = np.concatenate([g.flatten() for g in glcm_matrices])
    av = np.mean(all_values)
    d = np.std(all_values)
    return av, d


def plot_histograms(original, equalized, ax_orig, ax_eq):
    ax_orig.hist(original.ravel(), bins=256, color='black')
    ax_orig.set_title('Гистограмма яркости исходного L')
    ax_eq.hist(equalized.ravel(), bins=256, color='black')
    ax_eq.set_title('Гистограмма яркости контрастированного L')


def process_and_plot_color(path):
    # Загрузка
    bgr_orig, hsl_orig = load_image_hsl(path)

    # Контрастирование L канала
    hsl_eq = equalize_l_channel(hsl_orig)
    bgr_eq = hsl_to_bgr(hsl_eq)

    # Берём L каналы для анализа (8-бит)
    l_orig = hsl_orig[:, :, 1]
    l_eq = hsl_eq[:, :, 1]

    # Строим GLCM для L до и после
    glcm_orig = [compute_glcm(l_orig, distances[0], a) for a in angles_rad]
    glcm_eq = [compute_glcm(l_eq, distances[0], a) for a in angles_rad]

    av_orig, d_orig = compute_av_d(glcm_orig)
    av_eq, d_eq = compute_av_d(glcm_eq)

    # Визуализация
    fig, axs = plt.subplots(5, 4, figsize=(15, 15))
    fig.suptitle(f'Результаты для {path}', fontsize=16)

    # Исходное цветное изображение и контрастированное
    axs[0, 0].imshow(cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Исходное цветное изображение')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('Цветное после контрастирования L')
    axs[0, 1].axis('off')

    # Гистограммы яркости L
    plot_histograms(l_orig, l_eq, axs[0, 2], axs[0, 3])

    # Визуализация GLCM (логарифмически нормированные)
    for i, angle in enumerate(angles_deg):
        axs[1, i].imshow(log_normalize(glcm_orig[i]), cmap='gray')
        axs[1, i].set_title(f'GLCM исходное\nугол {angle}°')
        axs[1, i].axis('off')

        axs[2, i].imshow(log_normalize(glcm_eq[i]), cmap='gray')
        axs[2, i].set_title(f'GLCM контрастированное\nугол {angle}°')
        axs[2, i].axis('off')

    # Вывод признаков AV и D
    axs[3, 0].text(0, 0.5, f'AV исходное: {av_orig:.6f}\nD исходное: {d_orig:.6f}', fontsize=12)
    axs[3, 0].axis('off')

    axs[3, 1].text(0, 0.5, f'AV контрастированное: {av_eq:.6f}\nD контрастированное: {d_eq:.6f}', fontsize=12)
    axs[3, 1].axis('off')

    # Оставшиеся ячейки пустые
    for j in range(2, 4):
        axs[3, j].axis('off')
    for i in range(4, 5):
        for j in range(4):
            axs[i, j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Пути к изображениям — поменяйте на свои
paths = [
    "Photos/ph1.png",
    "Photos/ph2.png",
    "Photos/ph5.png"
]

for p in paths:
    process_and_plot_color(p)
