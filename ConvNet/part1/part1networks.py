import keras

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

def alexNetModel(input_shape,num_classes):
    model = Sequential()

    model.add(Conv2D(96,kernel_size=11,strides=4,input_shape=input_shape,activation="elu",padding='valid'))
    model.add(MaxPooling2D(pool_size=3,strides=2))

    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(256,kernel_size=5,strides=1,padding="same",activation="elu"))
    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(keras.layers.BatchNormalization())

    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu"))
                

    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu"))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu"))



    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Flatten())
    #model.add(keras.layers.Dropout(0.85))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2()))
    model.add(keras.layers.Dropout(0.5))
    model.add(Dense(4096, activation='elu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    return model


def alexExperimentNetModel(input_shape,num_classes,drop1,drop2,regularizerRate1,regularizerRate2,initializer):
    model = Sequential()

    model.add(Conv2D(96,kernel_size=5,strides=4,input_shape=input_shape,activation="elu",padding='valid',kernel_initializer=initializer,))
    model.add(MaxPooling2D(pool_size=3,strides=2))

    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(512,kernel_size=5,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(keras.layers.BatchNormalization())

    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
                

    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))



    #model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Flatten())
    #model.add(keras.layers.Dropout(0.85))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate1),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(keras.layers.Dropout(drop1))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate2),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(keras.layers.Dropout(drop2))
    model.add(Dense(1000, activation='relu',kernel_initializer=initializer,bias_initializer="zeros"))

    model.add(Dense(num_classes, activation='sigmoid'))

    return model

def alexExperimentNetModelDeeper(input_shape,num_classes,drop1,drop2,regularizerRate1,regularizerRate2,initializer):
    
    
    model = Sequential()

    model.add(Conv2D(64,kernel_size=11,strides=4,input_shape=input_shape,activation="elu",padding='valid',kernel_initializer=initializer,))
    #model.add(MaxPooling2D(pool_size=3,strides=2))

    #model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    #model.add(MaxPooling2D(pool_size=3,strides=2))


    #model.add(keras.layers.BatchNormalization())

   # model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
                

    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))



    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Flatten())
    #model.add(keras.layers.Dropout(0.85))
    #model.add(Dense(1096, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate1),kernel_initializer=initializer,bias_initializer="zeros"))
    #model.add(Dense(1096, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate1),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(Dense(296, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate1),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(Dense(1096, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate1),kernel_initializer=initializer,bias_initializer="zeros"))
    #model.add(Dense(256, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate1),kernel_initializer=initializer,bias_initializer="zeros"))
    #model.add(keras.layers.Dropout(drop1))
   # model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate2),kernel_initializer=initializer,bias_initializer="zeros"))
    #model.add(keras.layers.Dropout(drop2))
    model.add(Dense(1000, activation='relu',kernel_initializer=initializer,bias_initializer="zeros"))

    model.add(Dense(num_classes, activation='sigmoid'))

    return model

"""
def alexExperimentNetModelVGG19(input_shape,num_classes,drop1,drop2,initializer):
    model = Sequential()

    model.add(Conv2D(64,kernel_size=3,strides=1,input_shape=input_shape,activation="elu",padding='valid',kernel_initializer=initializer,))
    
    model.add(keras.layers.BatchNormalization())

    #model.add(Conv2D(64,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
    model.add(Conv2D(64,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))

    model.add(Conv2D(128,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
    model.add(Conv2D(128,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))

    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))
    


    model.add(Flatten())
    #model.add(keras.layers.Dropout(0.85))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2(),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(keras.layers.Dropout(drop1))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2(),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(keras.layers.Dropout(drop2))
    model.add(Dense(1000, activation='relu',kernel_initializer=initializer,bias_initializer="zeros"))

    model.add(Dense(num_classes, activation='softmax'))

    return model
"""

def leNet(input_shape,num_classes):
     model = keras.models.Sequential()
     model.add(keras.layers.Conv2D(6,kernel_size= (5,5),input_shape = input_shape))
     model.add(keras.layers.MaxPool2D())
     model.add(keras.layers.Conv2D(16,kernel_size=(5,5)))
     model.add(keras.layers.MaxPool2D())
     model.add(keras.layers.Dropout(0.25))
     model.add(keras.layers.Flatten())
     model.add(keras.layers.Dense(120,activation="relu"))
     model.add(keras.layers.Dense(400,activation="relu"))
     model.add(keras.layers.Dense(84,activation="relu"))
     model.add(keras.layers.Dropout(0.5))
     model.add(keras.layers.Dense(num_classes,activation="softmax"))
     return model
 
def leNetExp(input_shape,num_classes):
     model = keras.models.Sequential()
     if len(input_shape)>1:#used for different models when weights form pretrained are used or not 
         model.add(keras.layers.Conv2D(32,kernel_size= (10,10),input_shape = input_shape))
         model.add(keras.layers.MaxPool2D())
         model.add(keras.layers.Conv2D(64,kernel_size=(3,3)))
         model.add(keras.layers.MaxPool2D())
         model.add(keras.layers.Dropout(0.25))
         model.add(keras.layers.Flatten())
         model.add(keras.layers.Dense(64,activation="relu"))
         model.add(keras.layers.Dense(64,activation="relu"))
     else:
         model.add(keras.layers.Dense(512,activation="relu",input_dim = input_shape))
     model.add(keras.layers.Dense(32,activation="relu"))
     model.add(keras.layers.Dropout(0.5))
     model.add(keras.layers.Dense(num_classes,activation="softmax"))

     model.summary()
     return model
