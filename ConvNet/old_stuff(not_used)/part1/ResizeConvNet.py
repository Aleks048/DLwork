
from __future__ import print_function
import keras
from keras.backend.tensorflow_backend import set_session

from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt

from PIL import Image

import xlrd

import os
import numpy
import math

import tensorflow as tf


import gc
import json






def resizeTheInput(address,name,trainPercent,numOfImagesDigit,numOfImagesLetter,width,height,firstOrRest):
  
    countImages = 0


    #generate random numbers that will use for testing and check that uniformly distributed around both classes
    excelFile = xlrd.open_workbook('C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\Codes.xlsx')

    worksheet = excelFile.sheet_by_name(name)
    testRandomNum = []
    crackIsThereCount=numOfImagesDigit*numOfImagesLetter*(1-trainPercent)//2
    crackNotThereCount=numOfImagesDigit*numOfImagesLetter*(1-trainPercent)//2
    

    while (crackIsThereCount!=0) or (crackNotThereCount!=0):
        temp=int(numpy.random.randint(1,numOfImagesDigit*numOfImagesLetter,size=1))
        if worksheet.cell(temp+4,4).value!=xlrd.empty_cell.value:
            if temp not in testRandomNum:
                if crackIsThereCount!=0:
                    testRandomNum+=[temp]
                    crackIsThereCount-=1
        else:
            if temp not in testRandomNum:
                if crackNotThereCount!=0:
                    testRandomNum+=[temp]
                    crackNotThereCount-=1


    for j in range(numOfImagesLetter):
        for i in range(15):#15 is the number of images per letter
            z=i+1
            if firstOrRest:
                img = Image.open(address+name+"\\"+name+"_original\\"+str(i+1)+chr(ord('A')+j)+".jpg")
            else:
                if i<9:
                    img = Image.open(address+name+"\\"+name+"_original\\"+name+" "+chr(ord('A')+j)+"0"+str(i+1)+" R0 A"+".jpg")
                else:
                    img = Image.open(address+name+"\\"+name+"_original\\"+name+" "+chr(ord('A')+j)+str(i+1)+" R0 A"+".jpg")
            

           
            #width,height = img.size

            #print(name+" "+str(i+1)+chr(ord('A')+j)+" R0 A")

            if countImages not in testRandomNum:
                if i<9:
                    img.resize((width,height)).save(address+name+"\\"+name+"_resized\\train\\0rot\\"+chr(ord('A')+j)+"0"+str(i+1)+".jpg")
                else:
                    img.resize((width,height)).save(address+name+"\\"+name+"_resized\\train\\0rot\\"+chr(ord('A')+j)+str(i+1)+".jpg")
            else:
                if i<9:
                    img.resize((width,height)).save(address+name+"\\"+name+"_resized\\test\\0rot\\"+chr(ord('A')+j)+"0"+str(i+1)+".jpg")
                else:
                    img.resize((width,height)).save(address+name+"\\"+name+"_resized\\test\\0rot\\"+chr(ord('A')+j)+str(i+1)+".jpg")
            countImages+=1

            img.close()
    return testRandomNum
    

def createTheClassificationText(name,crackType,trainPercent,numOfImagesDigit,numOfImagesLetter,testrandomNumList):
    
    crackTrainFile = open("../resizedData/"+name+"/trainCrack"+crackType+".txt","w")
    crackTestFile =  open("../resizedData/"+name+"/testCrack"+crackType+".txt","w")

    cracksTrainArr = numpy.array([])
    cracksTestArr = numpy.array([])

    excelFile = xlrd.open_workbook('C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\Codes.xlsx')

    worksheet = excelFile.sheet_by_name(name)
    
    count=0
    for i in range(4,109):
        if worksheet.cell(i,4).value!=xlrd.empty_cell.value:
            
            if count not in testrandomNumList:
                cracksTrainArr = numpy.append(cracksTrainArr,[1])
            else:
                #print(worksheet.cell(i,0).value)
                cracksTestArr = numpy.append(cracksTestArr,[1])
        else:
            if count not in testrandomNumList:
                cracksTrainArr = numpy.append(cracksTrainArr,[0])
            else:
                #print(worksheet.cell(i,0).value)
                cracksTestArr = numpy.append(cracksTestArr,[0])
        count+=1
     
    for i in cracksTrainArr:
        crackTrainFile.write(str(i)+"\n")
    for i in cracksTestArr:
        crackTestFile.write(str(i)+"\n")
    
    crackTrainFile.close()
    crackTestFile.close()
    
    return cracksTrainArr,cracksTestArr


def createNumpyArraysFromImages(directoryIn,img_x,img_y):
    x_train=numpy.empty((0,img_x,img_y,3))
    x_test=numpy.empty((0,img_x,img_y,3))

    #reflection

    #rotation
    x_trainRotated=numpy.empty((0,img_x,img_y,3))
    x_trainRotated180=numpy.empty((0,img_x,img_y,3))
    x_trainRotated270=numpy.empty((0,img_x,img_y,3))
    
    x_testRotated=numpy.empty((0,img_x,img_y,3))
    x_testRotated180=numpy.empty((0,img_x,img_y,3))
    x_testRotated270=numpy.empty((0,img_x,img_y,3))
    
    directoryTrain=os.fsencode(directoryIn+"\\train\\0rot\\")
 
    for file in os.listdir(directoryTrain):
        fileName = os.fsencode(file)
       
        if str(fileName)[2:-1]!='numpyArrays' :
        
            tempArr = numpy.asarray(Image.open(directoryIn+"\\train\\0rot\\"+str(fileName)[2:-1],mode='r'))
            
            #x_train=numpy.append(x_train,[tempArr],axis=0)
        
            #rotations arrays
            rotated90 = numpy.rot90(tempArr,axes=(-3,-2))
            rotated180 = numpy.rot90(rotated90,axes=(-3,-2))     
            rotated270 = numpy.rot90(rotated180,axes=(-3,-2))
            
            #reflections arrays
            reflection0 = numpy.fliplr(tempArr)
            reflection90 = numpy.fliplr(rotated90)
            reflection180 = numpy.fliplr(rotated180)
            reflection270 = numpy.fliplr(rotated270)
            


            #creating rotated images
            img=Image.fromarray(tempArr,'RGB')
            img90=Image.fromarray(rotated90,'RGB')
            img180=Image.fromarray(rotated180,'RGB')
            img270=Image.fromarray(rotated270,'RGB')
            #creating reflection images
            refl0 = Image.fromarray(reflection0,"RGB")
            refl90 = Image.fromarray(reflection90,"RGB")
            refl180 = Image.fromarray(reflection180,"RGB")
            refl270 = Image.fromarray(reflection270,"RGB")

            #numpy arrays save
            numpy.save(directoryIn+"\\train\\0rot\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl0)
            numpy.save(directoryIn+"\\train\\90rot\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl90)
            numpy.save(directoryIn+"\\train\\180rot\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl180)
            numpy.save(directoryIn+"\\train\\270rot\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl270)
            
            numpy.save(directoryIn+"\\train\\0ref\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl0)
            numpy.save(directoryIn+"\\train\\90ref\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl90)
            numpy.save(directoryIn+"\\train\\180ref\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl180)
            numpy.save(directoryIn+"\\train\\270ref\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl270)

            #images save
            
            img90.save(directoryIn+"\\train\\90rot\\"+str(fileName)[2:-1])
            img180.save(directoryIn+"\\train\\180rot\\"+str(fileName)[2:-1])
            img270.save(directoryIn+"\\train\\270rot\\"+str(fileName)[2:-1])
            
            refl0.save(directoryIn+"\\train\\0ref\\"+str(fileName)[2:-1])
            refl90.save(directoryIn+"\\train\\90ref\\"+str(fileName)[2:-1])
            refl180.save(directoryIn+"\\train\\180ref\\"+str(fileName)[2:-1])
            refl270.save(directoryIn+"\\train\\270ref\\"+str(fileName)[2:-1])

            '''
            img.show()
            refl0.show()
            img90.show()
            refl90.show()
            img180.show()
            refl180.show()
            img270.show()
            refl270.show()
            print()
            '''

            

        
            
        
    
    directoryTest=os.fsencode(directoryIn+"\\test\\0rot\\")
    for file in os.listdir(directoryTest):
        fileName = os.fsencode(file)
        
        if str(fileName)[2:-1]!='numpyArrays' :
            file =Image.open(directoryIn+"\\test\\0rot\\"+str(fileName)[2:-1])
            tempArr = numpy.asarray(file)
            #x_test = numpy.append(x_test,[tempArr],axis=0)

            #reflection


            #rotation arrays
            rotated90 = numpy.rot90(tempArr,axes=(-3,-2))
            rotated180 = numpy.rot90(rotated90,axes=(-3,-2))
            rotated270 = numpy.rot90(rotated180,axes=(-3,-2))
            
            #reflections arrays
            reflection0 = numpy.fliplr(tempArr)
            reflection90 = numpy.fliplr(rotated90)
            reflection180 = numpy.fliplr(rotated180)
            reflection270 = numpy.fliplr(rotated270)


            #creating rotated images
            img90=Image.fromarray(rotated90,'RGB')
            img180=Image.fromarray(rotated180,'RGB')
            img270=Image.fromarray(rotated270,'RGB')
            #creating reflection images
            refl0 = Image.fromarray(reflection0,"RGB")
            refl90 = Image.fromarray(reflection90,"RGB")
            refl180 = Image.fromarray(reflection180,"RGB")
            refl270 = Image.fromarray(reflection270,"RGB")

            #numpy arrays save
            numpy.save(directoryIn+"\\test\\0rot\\numpyArrays\\"+str(fileName)[2:-5]+".npy",tempArr)
            numpy.save(directoryIn+"\\test\\90rot\\numpyArrays\\"+str(fileName)[2:-5]+".npy",rotated90)
            numpy.save(directoryIn+"\\test\\180rot\\numpyArrays\\"+str(fileName)[2:-5]+".npy",rotated180)
            numpy.save(directoryIn+"\\test\\270rot\\numpyArrays\\"+str(fileName)[2:-5]+".npy",rotated270)

            numpy.save(directoryIn+"\\test\\0ref\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl0)
            numpy.save(directoryIn+"\\test\\90ref\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl90)
            numpy.save(directoryIn+"\\test\\180ref\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl180)
            numpy.save(directoryIn+"\\test\\270ref\\numpyArrays\\"+str(fileName)[2:-5]+".npy",refl270)

            #images save
            img90.save(directoryIn+"\\test\\90rot\\"+str(fileName)[2:-1])
            img180.save(directoryIn+"\\test\\180rot\\"+str(fileName)[2:-1])
            img270.save(directoryIn+"\\test\\270rot\\"+str(fileName)[2:-1])
            
            refl0.save(directoryIn+"\\test\\0ref\\"+str(fileName)[2:-1])
            refl90.save(directoryIn+"\\test\\90ref\\"+str(fileName)[2:-1])
            refl180.save(directoryIn+"\\test\\180ref\\"+str(fileName)[2:-1])
            refl270.save(directoryIn+"\\test\\270ref\\"+str(fileName)[2:-1])

            file.close()



def getImages(directoryIn,folderName,img_x,img_y,starting_file_num,finish_file_num,kFoldImageNumList,excludeIncludeImageNumFromList):
    
    count = 0
    #print(directoryIn+folderName+"\\numpyArrays\\")
    directoryTrain=os.fsencode(directoryIn+folderName+"\\numpyArrays\\")
 
    x_out=numpy.empty((0,img_x,img_y,3))


    #print(directoryTrain)
    
    for file in os.listdir(directoryTrain):

        if ((count>=starting_file_num) and (count<finish_file_num)) :
            if (excludeIncludeImageNumFromList and (count not in kFoldImageNumbers)):
                fileName = os.fsencode(file)
                '''
                #with Image.open(directoryIn+folderName+"\\"+str(fileName)[2:-1]) as file:
                    #tempArr = numpy.asarray(file)
                    #img=Image.fromarray(tempArr,'RGB')
                    #img.show()
                '''
                tempArr = numpy.load(directoryIn+folderName+"\\numpyArrays\\"+str(fileName)[2:-1])
                x_out=numpy.append(x_out,[tempArr],axis=0)
            if (not(excludeIncludeImageNumFromList) and (count in kFoldImageNumbers)):
                fileName = os.fsencode(file)
                '''
                #with Image.open(directoryIn+folderName+"\\"+str(fileName)[2:-1]) as file:
                   # tempArr = numpy.asarray(file)
                    #img=Image.fromarray(tempArr,'RGB')
                    #img.show()
                '''
                tempArr = numpy.load(directoryIn+folderName+"\\numpyArrays\\"+str(fileName)[2:-1])
                x_out=numpy.append(x_out,[tempArr],axis=0)
                
        count+=1
    count=0   
    return x_out



def getNumpyArraysNames(directoryIn,folderName,trainTest):
    
    path = directoryIn+folderName+"numpyArrays\\"

    names_out=[]

    directoryTrain=os.fsencode(path)
 
    for file in os.listdir(directoryTrain):
        fileName = os.fsencode(file)
        tempStr=path+str(fileName)[2:-5]
        names_out+=[tempStr]
    
    names_out.sort()
    
    return names_out





img_x,img_y = 180, 180
numOfChannels = 3

dataAugTypes = ["0rot","90rot","180rot","270rot","0ref","90ref","180ref","270ref"]


batch_size = 95  # math.floor((numOfFolders*numOftrainImages*4)/40)

num_classes = 2
epochs = 3

numOfTrainImages =95
numOfTestImages= 10




#image preprocessing

#resize the input

y_train=numpy.empty(0)
y_test=numpy.empty(0)

x_train=numpy.empty((0, img_x, img_y, numOfChannels))
x_test=numpy.empty((0, img_x, img_y, numOfChannels))

#get the image data


foldersDone = [0,1,2,3,10,18,22]

numOfFolders = len(foldersDone)+1

for i in foldersDone:
    print("ho")
    
   
    #resizing the images and the augmented images
    """
    if i==0:
        randomTestNumList = resizeTheInput("C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\","AA"+chr(ord('A')+i),0.9,15,7,img_x,img_y,True)
    else:
        randomTestNumList = resizeTheInput("C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\","AA"+chr(ord('A')+i),0.9,15,7,img_x,img_y,False)
    
    with open("C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\AA"+chr(ord('A')+i)+"\\AA"+chr(ord('A')+i)+"_testNum.txt", 'w') as outfile:
        for z in randomTestNumList:
            outfile.write("%s " % z)
    outfile.close()

    #create the numpy image arrays
    createNumpyArraysFromImages("C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\AA"+chr(ord('A')+i)+"\\AA"+chr(ord('A')+i)+"_resized\\",img_x,img_y)    
    print()
    """

    tempStr=""
    with open("C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\AA"+chr(ord('A')+i)+"\\AA"+chr(ord('A')+i)+"_testNum.txt", 'r') as infile:
        for line in infile:    
            tempStr+=line
    testList=[]

    for k in tempStr.split():
        testList+=[int(k)]
    
    #create the classification file
    
    y_trainTemp,y_testTemp = createTheClassificationText("AA"+chr(ord('A')+i),"OCA",0.9,15,7,testList)


    y_train = numpy.append(y_train,y_trainTemp,axis=0)
    y_train = numpy.append(y_train,y_trainTemp,axis=0)
    y_train = numpy.append(y_train,y_trainTemp,axis=0)
    y_train = numpy.append(y_train,y_trainTemp,axis=0)
    y_train = numpy.append(y_train,y_trainTemp,axis=0)
    y_train = numpy.append(y_train,y_trainTemp,axis=0)
    y_train = numpy.append(y_train,y_trainTemp,axis=0)
    y_train = numpy.append(y_train,y_trainTemp,axis=0)

    y_test = numpy.append(y_test,y_testTemp,axis=0)
    y_test = numpy.append(y_test,y_testTemp,axis=0)
    y_test = numpy.append(y_test,y_testTemp,axis=0)
    y_test = numpy.append(y_test,y_testTemp,axis=0)
    y_test = numpy.append(y_test,y_testTemp,axis=0)
    y_test = numpy.append(y_test,y_testTemp,axis=0)
    y_test = numpy.append(y_test,y_testTemp,axis=0)
    y_test = numpy.append(y_test,y_testTemp,axis=0)
    



   
#create the neural network model



'''
img_x,img_y = 28,28 
numOfChannels=1
num_classes=10
(x_train,y_train),(x_test,y_test)=mnist.load_data()
'''


#reshaping the images
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, numOfChannels)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, numOfChannels)

input_shape = (img_x,img_y,numOfChannels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)





#to keep the history of epoch training
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
    
    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
    

history = AccuracyHistory()    






#train the model



#manually train the model
testResults =[]

NumOfK_fold = 6
kFoldLength = int(numOfTrainImages/NumOfK_fold)

numOfKFoldImages = 0



#dataGenerator
class DataGenerator(keras.utils.Sequence):
    def __init__(self,list_IDs,labels,batch_size=95,dim = (227,227),n_channels=3,n_classes=2,shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    def on_epoch_end(self):
        self.indexes = numpy.arange(len(self.list_IDs))
        if self.shuffle == True:
            numpy.random.shuffle(self.indexes)  
        
    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = numpy.empty((self.batch_size, *self.dim, self.n_channels))
      y = numpy.empty((self.batch_size), dtype=int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample

          X[i,] = numpy.load(ID + '.npy')

          #print(ID)
          # Store class
          y[i] = self.labels[ID]

      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
      
    def __len__(self):
         'Denotes the number of batches per epoch'
         return int(numpy.floor(len(self.list_IDs) // self.batch_size))

    def __getitem__(self, index):
          'Generate one batch of data'
          # Generate indexes of the batch
          indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
          
          # Find list of IDs
          list_IDs_temp = [self.list_IDs[k] for k in indexes]

          
          # Generate data
          X, y = self.__data_generation(list_IDs_temp)

          return X, y

#generator stuff

partition = {'train':[],"validation":[],"test":[]}

#train and test file names
train_data_names = []
test_data_names = []
for i in foldersDone:
    for j in dataAugTypes:
        letter = str(chr(ord('A')+i))
        temp = getNumpyArraysNames("C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\"+"AA"+letter+"\\"+"AA"+letter+"_resized\\train\\",j+"\\",True)
        train_data_names += temp
for i in foldersDone:
    for j in dataAugTypes:
        letter = str(chr(ord('A')+i))
        temp = getNumpyArraysNames("C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\"+"AA"+letter+"\\"+"AA"+letter+"_resized\\test\\",j+"\\",False)
        test_data_names += temp


labelsTrain = {}
labelsValidate = {}
labelsTest = {}
scores = []
epochs = 30#20#20#20



numSeeds=20
NumOfK_fold=5

numOfKFoldImages = int(len(train_data_names)/NumOfK_fold)
for _ in range(2):
    changeBatch = 26
    batches =[
        80,#300,
        26+i,26+i,26+i,52+i,52+i,52+i,52+i,78+i,78+i,78+i,78+i,104+i,104+i,104+i,104+i,130+i,130+i,130+i,130+i]#numpy.random.uniform(5,256,numSeeds) 
    alphas=[
            1.3/(pow(5,6)),#1.3/(pow(5,11)),#1.3/(pow(5,6)),
            1e-2,1e-3,1e-4,1e-1,1e-2,1e-3,1e-4,1e-1,1e-2,1e-3,1e-4,1e-1,1e-2,1e-3,1e-4,1e-1,1e-2,1e-3,1e-4]#numpy.random.uniform(1e-13,0.15,numSeeds)     #[0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015]    
    beta_ones=numpy.random.uniform(0.85,1.0,numSeeds)  
    beta_twos=numpy.random.uniform(0.95,1.0,numSeeds)  
    decays=numpy.random.uniform(1e-13,0.01,numSeeds)        
    epsilons=numpy.random.uniform(0.0,1e-9,numSeeds)
    dropouts1=numpy.random.uniform(0.0,1,numSeeds)
    dropouts2=numpy.random.uniform(0.0,1,numSeeds)

    count=0
    fileNum=0
    for k in range(numSeeds):
    
      
        tempScores=[]
        for j in range(NumOfK_fold):
                
                tf.reset_default_graph()
                gc.collect()

                
                config = tf.ConfigProto()  
                config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
                sess = tf.Session(config=config)  
                set_session(sess)  # set this TensorFlow session as the default session for Keras.
                


                params = {'dim': (img_x,img_y),
                          'batch_size': int(batches[k]),
                          'n_classes': num_classes,
                          'n_channels': numOfChannels,
                          'shuffle': True}


                #model = alexNet(dropout1=0.8,dropout2=0.8,fcSize1=4096,fcSize2=4096,fcSize3=1000)

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
                model.add(Dense(9096, activation='elu',kernel_regularizer=keras.regularizers.l2()))
                model.add(keras.layers.Dropout(0.92))
                model.add(Dense(4096, activation='elu'))
                model.add(keras.layers.Dropout(0.5))
                model.add(Dense(1000, activation='relu'))

                model.add(Dense(num_classes, activation='softmax'))

                
                print("seed: "+str(k+1)+" Fold :"+str(j+1))
        
                alpha = alphas[k]
                beta_1 = 0.9
                beta_2 = 0.999
                decay = alpha*2
                epsilon = 10e-8
                amsGrad = True
                print(alpha)
                #optimizer = keras.optimizers.SGD(lr =alpha,momentum = 0.99,nesterov =True)
                optimizer = keras.optimizers.Nadam(lr=alpha, beta_1=beta_1, beta_2=beta_2, epsilon=None, schedule_decay=decay)
                #optimizer=keras.optimizers.Adam(lr=alpha,beta_1=beta_1,beta_2=beta_2,decay=decay,epsilon = epsilon,amsgrad=amsGrad)
                
                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer = optimizer
                              ,metrics=['accuracy'])
                
                #if use cross vlidation
                partition["train"]=train_data_names[:j*numOfKFoldImages]+train_data_names[(j+1)*numOfKFoldImages if j!=NumOfK_fold-1 else len(train_data_names):]
                partition["validation"] = train_data_names[j*numOfKFoldImages:(j+1)*numOfKFoldImages if j!=NumOfK_fold-1 else len(train_data_names)]
                partition["test"] = test_data_names

        

                trainCount=0
                validationCount=0
                testCount=0

                #print(len(partition["train"]))
                #print(len(partition["validation"]))
                #print(len(y_train))

                for i in range(len(y_train)):
                    if not((i>=j*numOfKFoldImages) and (i < (j+1)*numOfKFoldImages if j!=NumOfK_fold-1 else len(train_data_names))):
                        labelsTrain[partition["train"][trainCount]] = 0 if y_train[i][0]==1 else 1
                        #print(partition["train"][trainCount] +" "+ str(y_train[i][1]),end="\n")
                        trainCount+=1
                    else:
                        labelsValidate[partition["validation"][validationCount]] = 0 if y_train[i][0]==1 else 1
                        #print(partition["validation"][validationCount] +" "+ str(y_train[i][1]),end="\n")
                        validationCount+=1
        
                #print(partition['train'])

                #the test values
                for a in partition["test"]:
                     labelsTest[a] = 0 if y_test[testCount][0]==1 else 1
                     #print(a)
                     testCount+=1
        
                #print(len(labelsValidate))
                #print(len(labelsTrain))
                #print(len(labelsTest))

                training_generator = DataGenerator(partition['train'], labelsTrain, **params)
                validation_generator = DataGenerator(partition['validation'], labelsValidate, **params)
                test_generator = DataGenerator(partition['test'], labelsTest, **params)


                tempHist = model.fit_generator(generator=training_generator,
                                    validation_data=validation_generator,
                                    validation_steps=int(len(partition["validation"])/batch_size) if int(len(partition["validation"])/batch_size)>1 else 1,
                                    epochs=epochs,
                                    use_multiprocessing=False,
                                    workers=1,
                                    max_queue_size=20,
                                    verbose=1
                                    #,callbacks=[history]
                                    )            
                
                
                score = model.evaluate_generator(generator = test_generator,
                                                 steps=int(len(partition["validation"])/batch_size) if int(len(partition["validation"])/batch_size)>1 else 1,
                                                 use_multiprocessing=False,
                                                 workers=1,
                                                 max_queue_size=20,
                                                 verbose=0
                                                 )
                

                print(score)
                
                temp={"loss":score[0],"validateAccuracy":score[1]}
                tempScores.append(temp.copy())

                del tempHist
                del score
                del model
                
                #sess.close()

                #for _ in range(15):
                gc.collect()

                tf.keras.backend.clear_session()
                keras.backend.clear_session()
                
  


        tempLoss=0
        tempAcc=0
        for i in tempScores:
            tempLoss+=i["loss"]
            tempAcc+=i["validateAccuracy"]
        tempLoss/=len(tempScores)
        tempAcc/=len(tempScores)
    
    
        tempDict = {"loss":tempLoss,"validateAccuracy":tempAcc,"batch":int(batches[k]),"alpha":alphas[k],"beta_one":beta_ones[k],"beta_two":beta_twos[k],"decay":decays[k],"epsilon":epsilons[k],
                    "dropout1":dropouts1[k],"dropour2": dropouts2[k]
                    }
   
        scores.append(temp.copy)

        if k%10==0:
           fileNum+=1
      
        myFile  = open("C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\temp_results\\scoresTemp"+str(fileNum)+".txt", "a") 


        myFile.write("Parameters of seed: "+str(count+1)+"\n\n")
        count+=1
        for d in tempDict:
        
            myFile.write("%s " % d )
            myFile.write("%s\n" %tempDict[d] )
        myFile.write("\n\n")
        myFile.close()
        print()
    



'''
print("\n the results")
for i in scores:
    for j in i:
        print(j+": "+str(i[j])+" ",end="")
    print()
'''
'''
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
'''
'''
print(len(testResults))
sumLoss=0
sumAcc=0
print(testResults)
'''

'''
for i in testResults:
    sumLoss+=i[0]
    sumAcc+=i[1]
loss = sumLoss/len(testResults)
acc = sumAcc/len(testResults)
print(' Total test loss:', loss)
print('Total test accuracy:', acc)
'''
'''
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
