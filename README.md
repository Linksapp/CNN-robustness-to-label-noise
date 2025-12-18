# CNN-robustness-to-label-noise

## Описание проекта

Этот проект представляет собой реализацию нейронной сети для классификации изображений из набора данных FashionMNIST. В работе исследуется влияние различных техник регуляризации на качество модели.

## Архитектура модели

Модель использует сверточную нейронную сеть (CNN) со следующей архитектурой:

1. **Сверточные слои:**
   - Conv2d(1, 32, 3, padding=1) + BatchNorm2d(32) + ReLU + MaxPool2d(2)
   - Conv2d(32, 64, 3, padding=1) + BatchNorm2d(64) + ReLU + MaxPool2d(2)
   - Conv2d(64, 128, 3, padding=1) + BatchNorm2d(128) + ReLU

2. **Полносвязные слои:**
   - Flatten()
   - Linear(6272, 512) + ReLU + Dropout
   - Linear(512, 128) + ReLU + Dropout
   - Linear(128, 10)

## Исследуемые техники регуляризации

Проект включает эксперименты с различными комбинациями техник регуляризации:

1. **Dropout** (0.3, 0.4, 0.5)
2. **L2 регуляризация** (0.0001, 0.001, 0.01, 0.05)
3. **Label smoothing** (0.1)
4. **Data augmentation** (углы поворота: 5°, 10°, 15°)
5. **Шум в метках** (10%, 20%, 30%)

## Настройки обучения

- **Оптимизатор:** SGD с learning rate = 0.001
- **Функция потерь:** CrossEntropyLoss
- **Размер батча:** 128
- **Количество эпох:** 100
- **Инициализация весов:** torch.manual_seed(15)

## Структура проекта
```
models/                          # Сохраненные модели
    CNN_10%_Dropout(0.3)_L2(0.0001)/
        metrics.csv              # Метрики обучения
        test_metrics.csv         # Метрики тестирования
        classes_accuracy.csv     # Точность по классам
        loss_fig.png             # График функции потерь
        accuracy_fig.png         # График точности
        eval.txt                 # Описание модели
        модель.pth               # Веса модели
```
## Использование

### Обучение модели:
```python
# Настройка параметров
noise_rate = 0
dropout_rate = 0
L2 = 0
label_smoothing_rate = 0
augm = 0

# Автоматическое создание имени модели
model_name = 'CNN'
if augm > 0: model_name += f'_augm{augm}'
if noise_rate > 0: model_name += f'_{int(noise_rate*100)}%'
if dropout_rate > 0: model_name += f'_Dropout({dropout_rate})'
if L2 > 0: model_name += f'_L2({L2})'
if label_smoothing_rate > 0: model_name += f'_label_smoothing({label_smoothing_rate})'

## Загрузка обученной модели:
```python
net = torch.load(f'models/{model_name}/{model_name}.pth', weights_only=False)
net.eval()
```
### Загрузка обученной модели:
```python
net = torch.load(f'models/{model_name}/{model_name}.pth', weights_only=False)
net.eval()
```
