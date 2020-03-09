import numpy as np
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,LambdaCallback
from tensorflow.keras.layers import LSTM,Dense,Input,Softmax,Convolution1D,Embedding,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model,Sequential
from utils import load,get_batch,predict_from_nothing,predict_from_head

UNITS = 256
batch_size = 64
epochs = 50
poetry_file = 'poetry.txt'

# 载入数据
x_data,char2id_dict,id2char_dict = load(poetry_file)
max_length = max([len(txt) for txt in x_data])
words_size = len(char2id_dict)

#-------------------------------#
#   建立神经网络
#-------------------------------#
inputs = Input(shape=(None,words_size))
x = LSTM(UNITS,return_sequences=True)(inputs)
x = Dropout(0.6)(x)
x = LSTM(UNITS)(x)                  
x = Dropout(0.6)(x)
x = Dense(words_size, activation='softmax')(x)
model = Model(inputs,x)

# model = Sequential([Input(shape=(None,words_size)),
#     LSTM(UNITS,return_sequences=True),
#     Dropout(0.6),
#     LSTM(UNITS),
#     Dropout(0.6),
#     Dense(words_size,activation="Softmax")])

#-------------------------------#
#   划分训练集验证集
#-------------------------------#
val_split = 0.1
np.random.seed(10101)
np.random.shuffle(x_data)
np.random.seed(None)
num_val = int(len(x_data)*val_split)
num_train = len(x_data) - num_val

#-------------------------------#
#   设置保存方案
#-------------------------------#
checkpoint = ModelCheckpoint('logs/loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

#-------------------------------#
#   设置学习率并训练
#-------------------------------#
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
for i in range(epochs):
    predict_from_nothing(i,x_data,char2id_dict,id2char_dict,model)
    model.fit_generator(get_batch(batch_size, x_data[:num_train], char2id_dict, id2char_dict),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=get_batch(batch_size, x_data[:num_val], char2id_dict, id2char_dict),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=1,
                    initial_epoch=0,
                    callbacks=[checkpoint])

#-------------------------------#
#   设置学习率并训练
#-------------------------------#
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy',
              metrics=['accuracy'])
        
for i in range(epochs):
    predict_from_nothing(i,x_data,char2id_dict,id2char_dict,model)
    model.fit_generator(get_batch(batch_size, x_data[:num_train], char2id_dict, id2char_dict),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=get_batch(batch_size, x_data[:num_train], char2id_dict, id2char_dict),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=1,
                    initial_epoch=0,
                    callbacks=[checkpoint])




