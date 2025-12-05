import os
import yaml
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Загрузка параметров
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


def get_model(model_name, num_classes):
    # 3 разные модели по заданию
    if model_name == "resnet18":
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:  # simple_cnn
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (params['base']['image_size'] // 4) ** 2, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    return model


def train():
    # Настройки MLflow
    mlflow.set_experiment("Food_Classification")

    img_size = params['base']['image_size']
    batch_size = params['base']['batch_size']
    train_dir = params['data']['current_train_data']  # Берем путь из конфига
    val_dir = params['data']['val_path']
    model_type = params['train']['model_type']
    epochs = params['train']['epochs']

    # Подготовка данных для PyTorch
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type, len(train_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['train']['learning_rate'])

    # ЗАПУСК MLflow RUN
    with mlflow.start_run():
        # Логируем параметры эксперимента
        mlflow.log_param("model", model_type)
        mlflow.log_param("data_source", train_dir)
        mlflow.log_param("epochs", epochs)

        print(f"Training {model_type} on {train_dir}...")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Валидация
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

            # Логируем метрики в MLflow
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("loss", running_loss / len(train_loader), step=epoch)

        # Сохраняем модель
        torch.save(model.state_dict(), "model.pth")
        mlflow.log_artifact("model.pth")


if __name__ == "__main__":
    train()