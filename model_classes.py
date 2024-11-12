import tensorflow as tf
from keras import layers, Sequential
import keras

@keras.utils.register_keras_serializable()
class SimpleNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # First Conv Layer
        self.add(layers.Conv2D(32, kernel_size=(5, 5), strides=2,
                               padding='same', activation='relu',
                               input_shape=input_shape,
                               kernel_initializer='he_normal'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

        # Second Conv Layer
        self.add(layers.Conv2D(64, kernel_size=(3, 3), strides=1,
                               padding='same', activation='relu',
                               kernel_initializer='he_normal'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

        # Third Conv Layer
        self.add(layers.Conv2D(128, kernel_size=(3, 3), strides=1,
                               padding='same', activation='relu',
                               kernel_initializer='he_normal'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

        # Flatten and a Single Fully Connected Layer
        self.add(layers.Flatten())
        self.add(layers.Dense(256, activation='relu'))
        self.add(layers.Dropout(0.5))  # Dropout to prevent overfitting
        self.add(layers.Dense(num_classes, activation='softmax'))

        # Compile the model
        self.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

@keras.utils.register_keras_serializable()
class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(layers.Conv2D(96, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'))
        self.add(layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None))

        self.add(layers.Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))
        self.add(layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)) 

        self.add(layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(layers.Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None))

        self.add(layers.Flatten())
        self.add(layers.Dense(4096, activation= 'relu'))
        self.add(layers.Dense(4096, activation= 'relu'))
        self.add(layers.Dense(1000, activation= 'relu'))
        self.add(layers.Dense(num_classes, activation= 'softmax'))

        self.compile(optimizer= tf.keras.optimizers.Adam(0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
