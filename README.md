## Computer Vision Задание 2
На основе ноутбука из предыдущего задания, свернуть нейросеть, используя минимум два уровня свертывания.

Слои в нашей модели.
```python
model = keras.Sequential([
                          keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
                          keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
                          keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                          keras.layers.Conv2D(128, (3, 3), padding='same', activation="relu"),
                          keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                          keras.layers.Conv2D(128, (3, 3), padding='same', activation="relu"),
                          keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                          keras.layers.Flatten(),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation="softmax")
])
```
Параметры компиляции модели:
```python
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
Результаты обучения модели:
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
```
<code>313/313 [==============================] - 21s 67ms/step - loss: 1.7729 - accuracy: 0.7199
Test loss: 1.7728607654571533
Test accuracy: 0.7199000120162964
</code>

![](https://github.com/SAJQDQ/CV3/blob/main/CV3%20png/Screenshot_5.png)  
![](https://github.com/SAJQDQ/CV3/blob/main/CV3%20png/Screenshot_1.png)  
![](https://github.com/SAJQDQ/CV3/blob/main/CV3%20png/Screenshot_2.png)  