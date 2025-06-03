from PIL import Image, ImageOps

def crop_and_convert_to_mono(input_path, output_path):
    # Загружаем изображение
    img = Image.open(input_path)

    # Конвертируем в оттенки серого, если это не так
    if img.mode != 'L':
        img = img.convert('L')

    # Инвертируем цвета для поиска bbox
    inverted_img = ImageOps.invert(img)

    # Находим ограничивающий прямоугольник вокруг текста
    bbox = inverted_img.getbbox()

    if bbox:
        cropped_img = img.crop(bbox)
    else:
        cropped_img = img  # если bbox не найден - оставляем как есть

    # Преобразуем в монохромный режим с порогом 128
    threshold = 128
    mono_img = cropped_img.point(lambda p: 255 if p > threshold else 0, '1')

    # Сохраняем результат
    mono_img.save(output_path)
    print(f"Сохранено обрезанное монохромное изображение: {output_path}")

# Вызов функции
crop_and_convert_to_mono('../Photo/original_images/phrase.png', '../Photo/mono&cut/phrase.bmp')
