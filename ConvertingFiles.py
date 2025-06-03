from PIL import Image

files = ['ph5']

for name in files:
    filename = 'Lab2/Photos/' + name + '/' + name + '.jpg'

    with Image.open(filename) as img:
        img.save('Lab2/Photos/' + name + '/' + name + '.bmp')

        print(f"{name}: {img.mode}")
