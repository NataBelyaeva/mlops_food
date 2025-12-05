import os
import yaml
import torch
from torchvision import transforms
from PIL import Image

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


def augment_data():
    input_path = params['data']['train_path']
    output_path = params['data']['augmented_path']

    # Определяем аугментации (повороты, флипы, цвет)
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # Всегда переворачиваем для новых данных
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Проходим по всем папкам и файлам
    for root, dirs, files in os.walk(input_path):
        for file in files:
            # Путь к исходному файлу
            src_file_path = os.path.join(root, file)

            # Структура папок в output
            relative_path = os.path.relpath(root, input_path)
            target_dir = os.path.join(output_path, relative_path)
            os.makedirs(target_dir, exist_ok=True)

            # 1. Копируем оригинал
            img = Image.open(src_file_path).convert("RGB")
            img.save(os.path.join(target_dir, file))

            # 2. Создаем и сохраняем аугментированную копию
            aug_img = augment_transform(img)
            aug_img.save(os.path.join(target_dir, f"aug_{file}"))

    print(f"Augmentation completed. Dataset saved to {output_path}")


if __name__ == "__main__":
    augment_data()