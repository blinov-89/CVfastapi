# CVfastapi
____
classification and segmentation image
____
## КЛАССИФИКАЦИЯ И СЕГМЕНТАЦИЯ ОБЪЕКТОВ С ИСПОЛЬЗОВАНИЕМ НЕЙРОСЕТЕЙ

9000 изображений

### Архитектура EfficientNetB0

При классификации используется предобученная model EfficientNetB0 с весами imagenet, входные размеры изображений 224х224х3

Данные поделены на train, test, validation выборки:

train = 6000 test = 1500 validation = 1500

- model.compile(loss='categorical_crossentropy',               

- optimizer=optimizers.RMSprop(learning_rate=1e-4),               

- metrics=['acc'])

### loss: 0.2158 acc: 0.9297

Разметка данных для обучения модели При разметка данных использовали VGG image annotation 

Разметка: Товар Ценник Цена Cохранение в разных форматах для загрузки данных

Архитектура модели в стиле U-Net Xception Получение предсказания расположения товара, ценника и цены товара
____

### Реализация с помощью FASTAPI
