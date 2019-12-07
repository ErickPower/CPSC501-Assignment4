import tensorflow as tf

print("--Get data--")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("--Process data--")
x_train, x_test = x_train / 255.0, x_test / 255.0


print("--Make model--")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(500, activation='relu'),
  tf.keras.layers.Dense(300, activation = 'relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

optim = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
history = model.fit(x_train, y_train, epochs=15, verbose=2)

print(f"Train Accuracy: {history.history.get('accuracy')[14]*100:.2f}")

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

model.save("MNIST.h5")