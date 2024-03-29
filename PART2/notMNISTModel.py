import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train, x_test = x_train.reshape(x_train.shape[0], 28,28,1), x_test.reshape(x_test.shape[0], 28,28,1)
 
print("--Make model--")
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16,3, input_shape=(28,28,1)),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(32,3),
  tf.keras.layers.MaxPool2D((2,2)),
  #tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(600, activation='relu'),
  tf.keras.layers.Dense(400, activation='relu'),
  tf.keras.layers.Dense(200, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

optim = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


print("--Fit model--")
model.fit(x_train, y_train, epochs=30, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2) 
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

model.save("notMNISTopt.h5")