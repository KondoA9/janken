import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

def winning_hand(hand):
    my_hand = max(hand)
    if my_hand == hand[0]:
        return [0,0,1]
    if my_hand == hand[1]:
        return [1,0,0]
    if my_hand == hand[2]:
        return [0,1,0]

def opponent_hand():
    return np.random.permutation([1,0,0])

def array_to_hand_name(array):
    arr = array[0]
    arr_max = max(arr)
    if arr_max == arr[0]:
        return "rock"
    elif arr_max == arr[1]:
        return "scissors"
    elif arr_max == arr[2]:
        return "paper"
    return ""


inputs = np.array([])
labels = np.array([])
inputs_test = np.array([])
labels_test = np.array([])
DATA_NUM = 10000

for i in range(DATA_NUM):
    hand = opponent_hand()
    inputs = np.append(inputs,hand)
    labels = np.append(labels,winning_hand(hand))
    hand = opponent_hand()
    inputs_test = np.append(inputs_test,hand)
    labels_test = np.append(labels_test,winning_hand(hand))

inputs = np.reshape(inputs,(DATA_NUM,3))
labels = np.reshape(labels,(DATA_NUM,3))
inputs_test = np.reshape(inputs_test,(DATA_NUM,3))
labels_test = np.reshape(labels_test,(DATA_NUM,3))


log_dir="C:\\tf_tmp\\janken\\"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_grads=True)

"""
model = keras.Sequential()
model.add(keras.layers.Dense(64,input_shape=(3,)))
model.add(keras.layers.Dense(64))
model.add(keras.layers.Dense(3))
model.add(keras.layers.Activation('softmax'))
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(inputs, labels, epochs=20, batch_size=500,callbacks=[tensorboard_callback],validation_data=(inputs_test, labels_test))
"""

inputs_layer=Input(shape=(3,))
x=Dense(64)(inputs_layer)
x=Dense(64)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=inputs_layer, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(inputs, labels) 


while(1):
    print("1: rock, 2: scissors, 3: paper")
    inp = input()
    x = []
    if int(inp) == 1:
        x = [1,0,0]
    elif int(inp) == 2:
        x = [0,1,0]
    elif int(inp) == 3:
        x = [0,0,1]
    else:
        continue
    x = np.reshape(x,(1,3))
    print("\t",array_to_hand_name(x),end="")

    predict = model.predict(x)
    print(" <- ",array_to_hand_name(predict),predict)

    print("")