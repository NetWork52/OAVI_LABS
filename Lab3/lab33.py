import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

# === 1. Загрузка изображения ===
def load_image(path):
    img = Image.open(path)
    return np.array(img)

image_color = load_image('Photos/ph5/collage_niblack_k_values.bmp')

# === 2. Перевод в полутоновое ===
def rgb_to_gray(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

image_gray = rgb_to_gray(image_color)

# === 3. Создание структурирующего элемента-диска 5×5 ===
def create_disk_kernel(size=5):
    y, x = np.ogrid[-2:3, -2:3]
    return (x**2 + y**2 <= 2.5**2).astype(np.uint8)

kernel = create_disk_kernel()

# === 4. Морфологическое открытие (полутоновое) ===
def morphological_open(img, kernel):
    eroded = ndimage.grey_erosion(img, structure=kernel)
    opened = ndimage.grey_dilation(eroded, structure=kernel)
    return opened

opened_gray = morphological_open(image_gray, kernel)

# === 5. Разностное изображение ===
diff = np.abs(image_gray.astype(np.int16) - opened_gray.astype(np.int16)
diff = np.clip(diff, 0, 255).astype(np.uint8)

# === 6. Морфологическая фильтрация каждого канала цветного изображения ===
def open_color_image(img, kernel):
    channels = [img[...,i] for i in range(3)]
    opened_channels = [morphological_open(ch, kernel) for ch in channels]
    return np.stack(opened_channels, axis=-1)

image_opened_color = open_color_image(image_color, kernel)

# === 7. Сохранение результатов ===
def save_image(array, path, color_mode='L'):
    if color_mode == 'L' and len(array.shape) == 2:
        Image.fromarray(array, mode=color_mode).save(path)
    else:
        Image.fromarray(array, mode='RGB').save(path)

save_image(image_gray, 'Photos/ph5/Results/collage_niblack_k_values_gray.bmp')
save_image(opened_gray, 'Photos/ph5/Results/collage_niblack_k_values_opened_gray.bmp')
save_image(diff, 'Photos/ph5/Results/collage_niblack_k_values_diff.bmp')
save_image(image_opened_color, 'Photos/ph5/Results/collage_niblack_k_values_opened_color.bmp')

# === 8. Визуализация результатов ===
titles = [
    'Исходное (цветное)',
    'Полутоновое изображение',
    'После открытия (полутон)',
    'Разностное изображение',
    'После открытия (цветное)'
]
images = [image_color, image_gray, opened_gray, diff, image_opened_color]

plt.figure(figsize=(20, 8))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig('00_collage.png', bbox_inches='tight')
plt.close()