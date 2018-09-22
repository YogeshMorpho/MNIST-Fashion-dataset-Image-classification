import numpy as np

from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam



(train_X,train_Y),(test_X,test_Y)=fashion_mnist.load_data()

train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)

train_X=train_X/255
test_X=test_X/255

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

batch_size=512
epochs=10
num_classes=10

fashion_model=Sequential([
Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=(28,28,1)),
MaxPooling2D(pool_size=2),
Dropout(0.2),

Flatten(),
Dense(32,activation='relu'),
Dense(10,activation='softmax')
])

fashion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

fashion_model.summary()

fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))





