
#learning without generator

if False:
    for k in range(epochs):
        tempTestResults = []
        print('Epoch: '+str(k+1))

        for fold in range(NumOfK_fold):
            y_train_start=0
            y_test_start=0

            print("Fold "+str(fold+1))

            kFoldImageNumbers = []

            #was used when randomly generated images for validation
            '''
            for i in range(numOfKFoldImages):
            
                temp = int(numpy.random.uniform(0,numOfTrainImages))
                while(temp in kFoldImageNumbers):
                    temp = int(numpy.random.uniform(0,numOfTrainImages))
                kFoldImageNumbers.append(temp)
               #kFoldImageNumbers.sort()#used when choose random for sampling 
            '''
            #get the image numbers that will be ussed for k-folding
            if fold!=NumOfK_fold-1:
                kFoldImageNumbers+=list(range(fold*kFoldLength,(fold+1)*kFoldLength))
                numOfKFoldImages =  int(numOfTrainImages*(1/NumOfK_fold))        
            if fold==NumOfK_fold-1:
                kFoldImageNumbers+=list(range(fold*kFoldLength,numOfTrainImages))
                numOfKFoldImages = numOfTrainImages-int(numOfTrainImages*(1/NumOfK_fold))*(fold-1)


            print(kFoldImageNumbers)


            localY_train = y_train
            localY_train_toAdd = y_train

            localY_Kfold = numpy.empty((0,2))



            #split the data into train and k-fold
            removeCount = 0
            for i in range(len(foldersDone)):
                for a in range(len(dataAugTypes)):
                    for j in kFoldImageNumbers:
                        localY_Kfold = numpy.append(localY_Kfold,[localY_train_toAdd[i*len(dataAugTypes)*numOfTrainImages+a*numOfTrainImages+j]],axis=0)
                        localY_train = numpy.delete(localY_train,i*len(dataAugTypes)*numOfTrainImages+a*numOfTrainImages+j-removeCount,axis = 0)
                        removeCount +=1 
        
            print(localY_Kfold.shape)

            for l in foldersDone:
                folderName = "AA"+chr(ord('A')+l)
                for j in dataAugTypes:
           
                    directory = "C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\"
                    for i in range(int(numOfTrainImages/batch_size)):
                        x_train = getImages(directory+folderName+"\\"+folderName+"_resized\\train\\",
                                            j,
                                            img_x,img_y,

                                            starting_file_num = i*batch_size,
                                            finish_file_num = min(i*batch_size+batch_size,numOfTrainImages),

                                            kFoldImageNumList = kFoldImageNumbers,
                                            excludeIncludeImageNumFromList = True)


                        model.train_on_batch(x_train,localY_train[y_train_start:y_train_start+x_train.shape[0]])
                    
                        y_train_start+=x_train.shape[0]
    
       
            y_Kfold_start = 0
            foldResults = []
            for l in foldersDone:
                folderName = "AA"+chr(ord('A')+l)
                for j in dataAugTypes:
            

                    directory = "C:\\Users\\ytr16\\source\\repos\\ConvNet\\resizedData\\"
                    for i in range(int(numOfTrainImages/batch_size)):

                        x_kFold = getImages(directory+folderName+"\\"+folderName+"_resized\\train\\",
                                            j,
                                            img_x,img_y,
                                        
                                            starting_file_num = i*batch_size,
                                            finish_file_num =  min(i*batch_size+batch_size,numOfTrainImages),
                                        
                                            kFoldImageNumList = kFoldImageNumbers,
                                            excludeIncludeImageNumFromList=False)
                        #print(x_kFold.shape)
                    
                        if x_kFold.shape[0]!=0:
                            test  = model.test_on_batch(x_kFold,localY_Kfold[y_Kfold_start:y_Kfold_start+x_kFold.shape[0]])
                            y_Kfold_start+=x_kFold.shape[0]
                            #print(test)
                            foldResults.append(test)
                            tempTestResults.append(test)
                
            sumLoss=0
            sumAcc=0
            #print(tempTestResults)
    
            for i in foldResults:
                sumLoss+=i[0]
                sumAcc+=i[1]
            loss = sumLoss/len(foldResults)
            acc = sumAcc/len(foldResults)
            print('Fold'+str(fold+1)+' test loss: ', loss)
            print('Fold'+str(fold+1)+' test accuracy: ', acc)


        #print(tempTestResults)
        sumLoss=0
        sumAcc=0
        #print(tempTestResults)
    
        for i in tempTestResults:
            sumLoss+=i[0]
            sumAcc+=i[1]
        loss = sumLoss/len(tempTestResults)
        acc = sumAcc/len(tempTestResults)
        print('Epoch '+str(k+1)+' test loss: ', loss)
        print('Epoch '+str(k+1)+' test accuracy: ', acc)

        testResults.append(tempTestResults)

#old alexnet
'''
#alexNet
def alexNet(dropout1,dropout2,fcSize1,fcSize2,fcSize3):
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

    #added for the sake
    #model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="relu"))
    #model.add(keras.layers.BatchNormalization())


    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Flatten())
    #model.add(keras.layers.Dropout(dropout1))
    model.add(Dense(fcSize1, activation='relu'))
    model.add(keras.layers.Dropout(dropout2))
    model.add(Dense(fcSize2, activation='relu'))
    model.add(keras.layers.Dropout(dropout1))
    model.add(Dense(fcSize3, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    #model.summary()
    return model
'''

#vgg16 net

def vgg16(dropout1,dropout2,fcSize1,fcSize2,fcSize3):
    model = Sequential()

    model.add(Conv2D(64,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(Conv2D(64,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(MaxPooling2D(pool_size=2,strides=2))
    
    model.add(Conv2D(128,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(Conv2D(128,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(MaxPooling2D(pool_size=2,strides=2))
  
    model.add(Conv2D(256,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(Conv2D(256,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(Conv2D(256,kernel_size=1,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(MaxPooling2D(pool_size=2,strides=2))

    model.add(Conv2D(512,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(Conv2D(512,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(Conv2D(512,kernel_size=1,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(MaxPooling2D(pool_size=2,strides=2))
   
    model.add(Conv2D(512,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(Conv2D(512,kernel_size=3,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(Conv2D(512,kernel_size=1,strides=1,input_shape=input_shape,activation="relu",padding='valid'))
    model.add(MaxPooling2D(pool_size=2,strides=2))

    model.add(keras.layers.BatchNormalization())


    model.add(Flatten())
    model.add(keras.layers.Dropout(dropout1))
    model.add(Dense(fcSize1, activation='relu'))
    model.add(keras.layers.Dropout(dropout2))
    model.add(Dense(fcSize2, activation='relu'))

    model.add(Dense(fcSize3, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    #model.summary()
    return model

#convonlynet


def convOnlyNet(dropout1,dropout2,fcSize1,fcSize2,fcSize3):
    model = Sequential()

    model.add(Conv2D(96,kernel_size=110,strides=10,input_shape=input_shape,activation="elu",padding='valid'))
    model.add(MaxPooling2D(pool_size=11,strides=2))

    model.add(Conv2D(96,kernel_size=11,strides=4,input_shape=input_shape,activation="elu",padding='valid'))
    model.add(MaxPooling2D(pool_size=3,strides=2))

    model.add(keras.layers.BatchNormalization())

    model.add(Conv2D(256,kernel_size=5,strides=1,padding="same",activation="elu"))
    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(keras.layers.BatchNormalization())

    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu"))
    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu"))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu"))

    #added for the sake
    #model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="relu"))



    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Flatten())
    model.add(keras.layers.Dropout(dropout1))
    model.add(Dense(fcSize1, activation='relu'))
    model.add(keras.layers.Dropout(dropout2))
    model.add(Dense(fcSize2, activation='relu'))

    model.add(Dense(fcSize3, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model
