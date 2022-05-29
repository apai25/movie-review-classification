import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM


dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(
    BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

model = Sequential()
model.add(encoder)
model.add(Embedding(input_dim=len(encoder.get_vocabulary()),
          output_dim=64, mask_zero=True))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_freq=1, validation_steps=30)

SAVE_FILEPATH = './model/saved_model'
tf.keras.models.save_model(model, SAVE_FILEPATH)
