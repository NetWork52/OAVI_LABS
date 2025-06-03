import cv2
import matplotlib.pyplot as plt
from PIL import Image
# Загружаем изображение
img = Image.open('Photos/ph5/ph5.png')
img = np.array(img)  # конвертация в numpy array


# Выводим изображение
plt.imshow(img)
plt.title('Loaded Image')
plt.axis('off')  # Убираем оси
#plt.show()

# Выделяем компоненты RGB
R = img.copy()
R[:, :, 1] = 0  # Убираем G
R[:, :, 2] = 0  # Убираем B

G = img.copy()
G[:, :, 0] = 0  # Убираем R
G[:, :, 2] = 0  # Убираем B

B = img.copy()
B[:, :, 0] = 0  # Убираем R
B[:, :, 1] = 0  # Убираем G

# Сохраняем результаты в файлы
cv2.imwrite('Photos/ph5/R_channel.png', cv2.cvtColor(R, cv2.COLOR_RGB2BGR))
cv2.imwrite('Photos/ph5/G_channel.png', cv2.cvtColor(G, cv2.COLOR_RGB2BGR))
cv2.imwrite('Photos/ph5/B_channel.png', cv2.cvtColor(B, cv2.COLOR_RGB2BGR))

# Отображаем результат
fig, ax = plt.subplots(1, 4, figsize=(16, 6))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(R)
ax[1].set_title('Red Channel')
ax[2].imshow(G)
ax[2].set_title('Green Channel')
ax[3].imshow(B)
ax[3].set_title('Blue Channel')

# Убираем оси для красоты
for a in ax:
    a.axis('off')
#plt.show()

import numpy as np

# Восстанавливаем изображение из отдельных каналов R, G, B
restored_image = np.stack([R[:, :, 0], G[:, :, 1], B[:, :, 2]], axis=-1)

# Сохраняем и отображаем результат
cv2.imwrite('images/restored_image.png', cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR))

# Показываем результат
plt.figure(figsize=(6, 6))
plt.imshow(restored_image)
plt.title('Restored Image from R, G, B')
plt.axis('off')
#plt.show()



def rgb_to_hsi(image):
    # Нормализуем значения RGB к диапазону [0, 1]
    img = image.astype('float32') / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Яркость
    I = (R + G + B) / 3

    # Насыщенность
    min_val = np.minimum(np.minimum(R, G), B)
    S = 1 - (min_val / (I + 1e-10))  # Добавляем небольшое значение, чтобы избежать деления на 0

    # Оттенок
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10  # Добавляем небольшой сдвиг для избежания деления на 0
    theta = np.arccos(num / den) * (180 / np.pi)  # В градусах

    H = np.where(B > G, 360 - theta, theta)

    return H, S, I


# Конвертируем в HSI
H, S, I = rgb_to_hsi(img)

# Сохраняем яркостную компоненту
cv2.imwrite('images/I_component.png', (I * 255).astype('uint8'))

# Отображаем результат
plt.figure(figsize=(6, 6))
plt.imshow(I, cmap='gray')
plt.title('Intensity Component (I)')
plt.axis('off')
#plt.show()

def hsi_to_rgb(H, S, I):
    H = H * (np.pi / 180)  # Переводим в радианы
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

    # Для угла в секторе 0 - 120 градусов
    idx = (H >= 0) & (H < 2 * np.pi / 3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi / 3 - H[idx]))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    # Для угла в секторе 120 - 240 градусов
    idx = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H[idx] -= 2 * np.pi / 3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi / 3 - H[idx]))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    # Для угла в секторе 240 - 360 градусов
    idx = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    H[idx] -= 4 * np.pi / 3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi / 3 - H[idx]))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    # Собираем RGB-изображение и возвращаем в диапазон 0-255
    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255).astype('uint8')
    return rgb

# Инвертируем яркость
I_inverted = 1 - I

# Преобразуем обратно в RGB
result_rgb = hsi_to_rgb(H, S, I_inverted)

# Сохраняем и отображаем результат
cv2.imwrite('ph5/inverted_intensity.png', cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))

# Показываем результат
plt.figure(figsize=(6, 6))
plt.imshow(result_rgb)
plt.title('Inverted Intensity')
plt.axis('off')
#plt.show()

# Создаем сетку 2x4 для отображения всех изображений
fig, ax = plt.subplots(2, 4, figsize=(20, 10))

# Верхний ряд
ax[0, 0].imshow(img)
ax[0, 0].set_title('Original Image')

ax[0, 1].imshow(R)
ax[0, 1].set_title('Red Channel')

ax[0, 2].imshow(G)
ax[0, 2].set_title('Green Channel')

ax[0, 3].imshow(B)
ax[0, 3].set_title('Blue Channel')

# Нижний ряд
ax[1, 0].imshow(restored_image)
ax[1, 0].set_title('Restored Image')

ax[1, 1].imshow(I, cmap='gray')
ax[1, 1].set_title('Intensity Component')

ax[1, 2].imshow(result_rgb)
ax[1, 2].set_title('Inverted Intensity')

# Отключаем пустой слот для красоты
ax[1, 3].axis('off')

# Убираем оси
for a in ax.ravel():
    a.axis('off')

# Делаем аккуратные отступы между изображениями
plt.tight_layout()

# Показываем все изображения в одном окне
plt.show()



def resize_image_linear(img, scale):
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Создаём пустое изображение нужного размера
    resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            # Определяем координаты в исходном изображении
            src_x = x / scale
            src_y = y / scale

            x0 = int(src_x)
            x1 = min(x0 + 1, width - 1)
            y0 = int(src_y)
            y1 = min(y0 + 1, height - 1)

            # Определяем весовые коэффициенты для интерполяции
            dx = src_x - x0
            dy = src_y - y0

            # Выполняем билинейную интерполяцию
            top = (1 - dx) * img[y0, x0] + dx * img[y0, x1]
            bottom = (1 - dx) * img[y1, x0] + dx * img[y1, x1]
            pixel = (1 - dy) * top + dy * bottom

            resized[y, x] = pixel

    return resized

M = 1.2  # Укажи во сколько раз растянуть изображение
stretched_image = resize_image_linear(img, M)

# Сохраняем и выводим результат
#cv2.imwrite('ph5/stretched_image.png', cv2.cvtColor(stretched_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('Photos/ph5/stretched_image.png', cv2.cvtColor(stretched_image, cv2.COLOR_RGB2BGR))

# Показываем результат на общей сетке
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title('Original Image')

ax[1].imshow(stretched_image)
ax[1].set_title(f'Stretched Image (x{M})')

# Убираем оси
for a in ax:
    a.axis('off')

plt.show()

def downscale_image(img, scale):
    height, width, channels = img.shape
    new_height = int(height / scale)
    new_width = int(width / scale)

    # Пустое изображение меньшего размера
    downscaled = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            # Определяем границы блока
            y0 = int(y * scale)
            y1 = min(int((y + 1) * scale), height)
            x0 = int(x * scale)
            x1 = min(int((x + 1) * scale), width)

            block = img[y0:y1, x0:x1]
            downscaled[y, x] = block.mean(axis=(0, 1))

    return downscaled

N = 2  # во сколько раз сжать
compressed_image = downscale_image(img, N)
cv2.imwrite('Photos/ph5/compressed_image.png', cv2.cvtColor(compressed_image, cv2.COLOR_RGB2BGR))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title('Original Image')

ax[1].imshow(compressed_image)
ax[1].set_title(f'Compressed Image (/ {N})')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()


M = 2  # растяжение
N = 4  # сжатие
K = M / N

# 1. Сначала растягиваем
stretched = resize_image_linear(img, M)

# 2. Затем сжимаем
resampled_two_pass = downscale_image(stretched, N)

# Сохраняем результат
cv2.imwrite('Photos/ph5/resampled_two_pass.png', cv2.cvtColor(resampled_two_pass, cv2.COLOR_RGB2BGR))

# Отображаем результат
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(img)
ax[0].set_title('Original Image')

ax[1].imshow(stretched)
ax[1].set_title(f'Stretched (x{M})')

ax[2].imshow(resampled_two_pass)
ax[2].set_title(f'Resampled (K={K:.2f}, two passes)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

K = 0.5  # масштаб (например, уменьшение в 2 раза)
resampled_one_pass = resize_image_linear(img, K)

# Сохраняем и отображаем
cv2.imwrite('Photos/ph5/resampled_one_pass.png', cv2.cvtColor(resampled_one_pass, cv2.COLOR_RGB2BGR))

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(img)
ax[0].set_title('Original Image')

ax[1].imshow(resampled_one_pass)
ax[1].set_title(f'Resampled (K={K:.2f}, one pass)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 4, figsize=(24, 12))

# Верхняя строка
ax[0, 0].imshow(img)
ax[0, 0].set_title('Original')

ax[0, 1].imshow(stretched)
ax[0, 1].set_title(f'Stretched (x{M})')

ax[0, 2].imshow(compressed_image)
ax[0, 2].set_title(f'Compressed (/ {N})')

ax[0, 3].imshow(resampled_two_pass)
ax[0, 3].set_title(f'Resampled 2-pass (K={K:.2f})')

# Нижняя строка
ax[1, 0].imshow(resampled_one_pass)
ax[1, 0].set_title(f'Resampled 1-pass (K={K:.2f})')

# Пустые ячейки — можно использовать для других сравнений
for i in range(1, 4):
    ax[1, i].axis('off')

# Отключаем оси
for a in ax.ravel():
    a.axis('off')

plt.tight_layout()
plt.show()