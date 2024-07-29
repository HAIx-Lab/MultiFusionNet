# -*- coding: utf-8 -*-

from keras import layers
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, concatenate, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.inception_v3 import InceptionV3

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Lambda,Concatenate

batch_size = 32
img_height, img_width = 224, 224
input_shape = (img_height, img_width, 3)
epochs = 100
input_tensor = Input(shape = input_shape)

base_model1=ResNet50V2(input_shape= input_shape,weights='imagenet', include_top=False, input_tensor=input_tensor)
for layer in base_model1.layers:
  layer.trainable=False
base_model1.summary()

#a=base_model1.get_layer("conv5_block2_3_conv").outputconv3_block3_2_conv
a=base_model1.get_layer("conv3_block3_2_conv").output
a=MaxPooling2D()(a)
a=MaxPooling2D()(a)
b=base_model1.get_layer("conv4_block6_3_conv").output
c=base_model1.get_layer("conv3_block4_3_conv").output
c=MaxPooling2D()(c)
d=base_model1.get_layer("conv2_block3_3_conv").output
d=MaxPooling2D()(d)
d=MaxPooling2D()(d)
print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
abcd=concatenate([a,b,c,d], axis=-1)
print(abcd.shape)

y = base_model1.output
conc=concatenate([y,abcd], axis=-1)
conc=BatchNormalization()(conc)
conc=Conv2D(2048, (1,1), activation='relu')(conc)
conc = GlobalAveragePooling2D()(conc)
print(conc.shape)

base_model2=InceptionV3(input_shape= input_shape,weights='imagenet', include_top=False, input_tensor=input_tensor)
for layer in base_model2.layers:
  layer.trainable=False
base_model2.summary()

T=concatenate([conc,conc1], axis=-1)
print(T.shape)

def add_specific_features(inputs):
    feature1 = inputs[0]
    feature2 = inputs[1]
    added_features = feature1 + feature2
    return added_features

added_features = Lambda(add_specific_features)([conc, conc1])

from keras.layers import Conv2D, Dropout
print(added_features.shape)
T=Dropout(0.3)(T)

T = Dense(256, activation='relu')(added_features)
predictions1 = Dense(3, activation='softmax')(T)
model3 = Model(inputs=input_tensor,outputs=predictions1)
model3.summary()
model3.compile(loss='CategoricalCrossentropy',optimizer='adam',metrics=['acc'] )

print("Trainable Parameters:")
for layer in model3.trainable_weights:
    print(layer.name)

print(model3)

print("Trainable Layers:")
for layer in model3.layers:
    if layer.trainable:
        print(layer.name)
        print(layer.count_params())
        print('---')

print("Trainable Layers:")
for layer in model3.layers:
    if layer.trainable:
        print(layer.name)
        print(layer.output_shape)
        print(layer.count_params())
        print('---')

for layer in model3.layers:
    if layer.trainable:
        print(layer.name)
        print("Number of Parameters:", layer.count_params())
        if isinstance(layer, Conv2D):
            print("Kernel Size:", layer.kernel_size)
        print("Output Size:", layer.output_shape)
        print()

training_data_dir='path'
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical")
validation_data_dir='path'
validation_data_generator = ImageDataGenerator(rescale=1./255)
validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False)

# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi

!pip install psutil
!pip install humanize

import psutil
import humanize
import os



# XXX: only one GPU on Colab and isn’t guaranteed

def printm():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))

printm()

from keras.callbacks import ModelCheckpoint
filepath = "path"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min', save_freq = 'epoch' )

#Training
H = model3.fit(
    training_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[checkpoint])

model3.save('path' )

