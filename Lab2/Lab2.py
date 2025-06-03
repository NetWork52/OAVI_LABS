from PIL import Image
import numpy as np

# Загрузка изображения
image = Image.open('Photos/ph5/ph5.bmp')
image = image.convert('RGB')  # На всякий случай убедимся, что это RGB

# Преобразование в массив numpy
pixels = np.array(image)

# Применение взвешенного среднего
gray_pixels = (0.299 * pixels[:, :, 0] +
               0.587 * pixels[:, :, 1] +
               0.114 * pixels[:, :, 2]).astype(np.uint8)

# Преобразуем обратно в изображение (одноканальное)
gray_image = Image.fromarray(gray_pixels, mode='L')

# Сохраняем результат
gray_image.save('Photos/ph5/ph5_gray.bmp')

from PIL import Image
import numpy as np

# Загрузка изображения
gray_image = Image.open('Photos/ph5/ph5_gray.bmp')
gray = np.array(gray_image).astype(np.float32)

# Размер окна и параметр k
window_size = 3
k = -0.2
pad = window_size // 2

# Паддинг изображения (отражением краёв)
padded = np.pad(gray, pad_width=pad, mode='reflect')

# Функции для локального среднего и std
def compute_local_stats(image, window):
    h, w = image.shape
    mean = np.zeros((h, w))
    std = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            region = image[i:i+window, j:j+window]
            mean[i, j] = np.mean(region)
            std[i, j] = np.std(region)
    return mean, std

mean, std = compute_local_stats(padded, window_size)

# Порог по Ниблэку
threshold = mean + k * std
threshold = threshold[pad:-pad, pad:-pad]  # убираем края, чтобы размер совпадал

# Бинаризация
binary = (gray >= threshold).astype(np.uint8) * 255
# Сохранение
binary_image = Image.fromarray(binary, mode='L')
binary_image.save('Photos/ph5/ph5_niblack_3x3.bmp')

from PIL import Image
import numpy as np

# Загрузка полутонового изображения
gray_image = Image.open('Photos/ph5/ph5_gray.bmp')
gray = np.array(gray_image).astype(np.float32)

# Параметры
window_size = 3
pad = window_size // 2
padded = np.pad(gray, pad_width=pad, mode='reflect')

def compute_local_stats(image, window):
    h, w = image.shape
    mean = np.zeros((h, w))
    std = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            region = image[i:i+window, j:j+window]
            mean[i, j] = np.mean(region)
            std[i, j] = np.std(region)
    return mean, std

mean, std = compute_local_stats(padded, window_size)

# Список значений k для экспериментов
k_values = [-0.1, -0.2, -0.3, -0.5]
images = []

for k in k_values:
    threshold = mean + k * std
    threshold = threshold[pad:-pad, pad:-pad]
    binary = (gray >= threshold).astype(np.uint8) * 255
    result = Image.fromarray(binary.astype(np.uint8), mode='L')
    result_path = f'Photos/ph5/niblack_k_{k:.1f}.bmp'
    result.save(result_path)
    images.append(np.array(result))

# Создание коллажа
collage = np.hstack(images)
collage_image = Image.fromarray(collage.astype(np.uint8))
collage_image.save('Photos/ph5/collage_niblack_k_values.bmp')
