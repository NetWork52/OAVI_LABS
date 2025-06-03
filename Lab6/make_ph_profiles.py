import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Загружаем монохромное изображение
img = Image.open("original/cropped_mono.bmp").convert("1")
img_np = np.array(img)

# Инвертируем: чёрное = 1, белое = 0
binary = 1 - img_np // 255

# Горизонтальный профиль (по строкам)
horizontal_profile = np.sum(binary, axis=1)

# Вертикальный профиль (по столбцам)
vertical_profile = np.sum(binary, axis=0)

# Сохраняем горизонтальный профиль
plt.figure(figsize=(10, 4))
plt.bar(range(len(horizontal_profile)), horizontal_profile, color='black')
plt.title("Горизонтальный профиль")
plt.xlabel("Строка")
plt.ylabel("Чёрные пиксели")
plt.tight_layout()
plt.savefig("orig_mono_profile/horizontal_profile.png")
plt.close()

# Сохраняем вертикальный профиль
plt.figure(figsize=(10, 4))
plt.bar(range(len(vertical_profile)), vertical_profile, color='black')
plt.title("Вертикальный профиль")
plt.xlabel("Столбец")
plt.ylabel("Чёрные пиксели")
plt.tight_layout()
plt.savefig("orig_mono_profile/vertical_profile.png")
plt.close()

print("Горизонтальный и вертикальный профили сохранены.")
