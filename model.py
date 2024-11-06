import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Flatten
from keras._tf_keras.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") /255 #normalização
x_test = x_test.astype("float32") / 255

#categorizar os rotulos

y_train = to_categorical(y_train,10)
x_test = to_categorical(x_test),10

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs= 5, batch_size= 32)

score = model.evaluate(x_train, y_train)