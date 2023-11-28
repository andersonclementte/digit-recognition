import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)

x_train, x_test = train_images / 255.0, test_images / 255.0 #normalize [0, 255 -> 0, 1]

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(lr=0.001)
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

batch_size = 64
epochs = 5

model.fit(x_train, train_labels, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)
# model.save('nn', save_format='tf')
model.save('nn.h5')

print('Evaluate on test data')
model.evaluate(x_test, test_labels, batch_size=batch_size, verbose=2)

new_model = keras.models.load_model('nn.h5')
new_model.evaluate(x_test, test_labels, batch_size=batch_size, verbose=2)