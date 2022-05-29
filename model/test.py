from tensorflow.keras.models import load_model
import os
print(os.path.relpath(__file__))
model = load_model('model/saved_model')


print(model.predict([['This movie was absolutely amazing, and I would love to watch it once again. One of the greatest movies of all time.']]))