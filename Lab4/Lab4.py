import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# === 1. Загрузка изображения ===
image_path = 'Photos/ph1/ph1_gray.bmp'  # Укажи путь к изображению
image_color = Image.open(image_path).convert('RGB')
image_gray = image_color.convert('L')
image_gray_np = np.array(image_gray)

# === 2. Определение операторов Круна ===
Gx = np.array([[17, 61, 17],
               [0,  0,  0],
               [-17, -61, -17]], dtype=np.float32)

Gy = np.array([[17,  0, -17],
               [61,  0, -61],
               [17,  0, -17]], dtype=np.float32)

# === 3. Применение фильтров ===
grad_x = cv2.filter2D(image_gray_np.astype(np.float32), -1, Gx)
grad_y = cv2.filter2D(image_gray_np.astype(np.float32), -1, Gy)
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# === 4. Нормализация ===
def normalize(img):
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

grad_x_norm = normalize(grad_x)
grad_y_norm = normalize(grad_y)
grad_magnitude_norm = normalize(grad_magnitude)

# === 5. Бинаризация по порогу (подбирается вручную) ===
threshold = 35# Подбери опытным путём
binary_grad = (grad_magnitude_norm > threshold).astype(np.uint8) * 255

# === 6. Отображение всех результатов ===
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(image_color)
axs[0, 0].set_title("Исходное изображение")
axs[0, 0].axis('off')

axs[0, 1].imshow(image_gray, cmap='gray')
axs[0, 1].set_title("Полутоновое изображение")
axs[0, 1].axis('off')

axs[0, 2].imshow(grad_x_norm, cmap='gray')
axs[0, 2].set_title("Gx (нормализованное)")
axs[0, 2].axis('off')

axs[1, 0].imshow(grad_y_norm, cmap='gray')
axs[1, 0].set_title("Gy (нормализованное)")
axs[1, 0].axis('off')

axs[1, 1].imshow(grad_magnitude_norm, cmap='gray')
axs[1, 1].set_title("G (модуль градиента)")
axs[1, 1].axis('off')

axs[1, 2].imshow(binary_grad, cmap='gray')
axs[1, 2].set_title(f"Бинаризация (порог = {threshold})")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()
