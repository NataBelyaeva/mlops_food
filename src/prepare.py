import os
import shutil
import random
import yaml
from sklearn.model_selection import train_test_split

# Загружаем параметры
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


def prepare_data():
    random.seed(params['base']['random_state'])
    raw_path = params['data']['raw_dataset_path']
    train_path = params['data']['train_path']
    val_path = params['data']['val_path']

    # Очистка старых папок
    if os.path.exists(train_path): shutil.rmtree(train_path)
    if os.path.exists(val_path): shutil.rmtree(val_path)

    # Классы (Food, Non-Food)
    classes = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]

    for class_name in classes:
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)

        files = os.listdir(os.path.join(raw_path, class_name))
        # Простая разбивка 80/20
        train_files, val_files = train_test_split(files, test_size=0.2, random_state=params['base']['random_state'])

        for f in train_files:
            shutil.copy(os.path.join(raw_path, class_name, f), os.path.join(train_path, class_name, f))
        for f in val_files:
            shutil.copy(os.path.join(raw_path, class_name, f), os.path.join(val_path, class_name, f))

    print(f"Data prepared: Train and Val folders created in {params['data']['train_path']}")


if __name__ == "__main__":
    prepare_data()