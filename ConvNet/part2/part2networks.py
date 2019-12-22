import numpy as np


import keras
from numpy import shape,asarray,transpose,empty
from matplotlib import pyplot as plt
import CONSTANTS as CONST

# supporting
#data transform
def labelsToClassification(labels):
    print(len([i for i in labels if i<3.0]))
    print(len([i for i in labels if i>3.0]))
    return asarray([0 if i<3.0 else 1 for i in labels])
def to3Classes(labels):
    print(len([i for i in labels if i<1.5]))
    print(len([i for i in labels if (i>1.5) and (i<4.5)]))
    print(len([i for i in labels if i>4.5]))
    return asarray([0 if i<1.5 else (1 if (i>1.5) and (i<4.5) else 2) for i in labels])
#plot

#models
def findingModelWithAutokeras(DataX,LabelsY,testData,testLabels,categorizeLabels,lr):
    import autokeras as ak
    if categorizeLabels:
        LabelsY = labelsToClassification(LabelsY)
        testLabels = labelsToClassification(testLabels)
    
    DataX = np.swapaxes(DataX,0,1)
    DataX = np.swapaxes(DataX,1,4)
    DataX = np.squeeze(DataX)
    testData = np.swapaxes(testData,0,1)
    testData = np.swapaxes(testData,1,4)
    testData = np.squeeze(testData)

    print(shape(DataX))
    #print(DataX[0][0])
    
    model = ak.ImageClassifier(verbose=True,augment = True)
    model.fit(DataX,LabelsY,10*60*60)
    model.final_fit(DataX,LabelsY,testData,testLabels, retrain=True)
    #x_train = np.random.rand(100, 30, 30, 1)
    #x_val  = np.random.rand(70, 30, 30, 1)
    #y_train = np.ceil(np.random.rand(100))
    #y_val = np.ceil(np.random.rand(70))
    #clf = ak.ImageClassifier(verbose=True)
    #clf.fit(x_train, y_train)
    pass

def rawData(DataX,LabelsY,testData,testLabels,categorizeLabels,lr):
    from numpy import sum

    #transforming the data

    DataX = np.swapaxes(DataX,0,1)
    DataX = np.swapaxes(DataX,1,4)
    DataX = np.squeeze(DataX)
    testData = np.swapaxes(testData,0,1)
    testData = np.swapaxes(testData,1,4)
    testData = np.squeeze(testData)

    def summingData(data):
        out = []
        for c in data:
            out.append(asarray([sum(v) for _,v in enumerate(c)]))
        return transpose(out)
    def threchSummingData(data,threchold):
        out = []
        for c in data:
            out.append(asarray([sum([100 if v2>threchold else (-1 if v2>0.5 else -5) for i,v2 in enumerate(np.ndarray.flatten(v))]) for _,v in enumerate(c)]))
        return transpose(out)
    def fancyPooling(data,regionWidthHeight,functionOnRegion):
        out = []
        print(shape(data))
        for im in data:
            imagerapresentation = []
            for c in im:
                x=0
                y=0
                sh = shape(c[0])
                
                oneTypeOfCrackPedictionsPooled = np.empty((4,60//regionWidthHeight,60//regionWidthHeight,1))
                while x<sh[0]:
                    while y<sh[1]:
                        
                        subarray = c[x+regionWidthHeight][y+regionWidthHeight] 
                        
                        oneTypeOfCrackPedictionsPooled[x//regionWidthHeight][y//regionWidthHeight]=functionOnRegion(data = subarray)
                        y+=regionWidthHeight
                    x+=regionWidthHeight
                
                imagerapresentation.append(oneTypeOfCrackPedictionsPooled)
            out.append(imagerapresentation)
        print(shape(out))
        return transpose(out)
    def test(data):
        return sum(data)
    #DataX = fancyPooling(DataX,5,test)

    #labels to categotical data
    if categorizeLabels:
        LabelsY = keras.utils.to_categorical(labelsToClassification(LabelsY))
        testLabels = keras.utils.to_categorical(labelsToClassification(testLabels))

    #NN
    #parameters
    lr = lr/2.5

    batchSize = 900
    numEpochs = 30000

    
    #shuffling train data
    rng_state = np.random.get_state()
    np.random.shuffle(DataX)
    np.random.set_state(rng_state)
    np.random.shuffle(LabelsY)


    regularizationRates = [0.01,0.01]
    numOfDense1Units = 32
    numOfDense2Units = 16
    numOfDense3Units =8

    #initializer = keras.initializers.glorot_normal()
    #activation = "sigmoid"
    initializer = keras.initializers.he_normal()
    activation = "relu"
    dropoutRate =0.4

   

    #regularization = keras.regularizers.L1L2(l1=regularizationRates[0],l2=regularizationRates[1])
    regularization = keras.regularizers.l2(regularizationRates[1])

    

    print(shape(DataX[0]))
    input = keras.layers.Input(shape=(60,60,4),name="input")
    #x = keras.layers.AveragePooling3D((1,2,2))(input)
    x = keras.layers.Conv2D(4,(7,7),activation=activation,kernel_initializer=initializer,)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(4,(7,7),activation=activation,kernel_initializer=initializer,)(x)
    x = keras.layers.BatchNormalization()(x)
    conv1 = keras.layers.MaxPool2D((4,4))(x)
    x = keras.layers.Conv2D(4,(2,2),activation=activation,kernel_initializer=initializer,)(conv1)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(4,(2,2),activation=activation,kernel_initializer=initializer,)(x)
    x = keras.layers.BatchNormalization()(x)
    conv2 = keras.layers.MaxPool2D((2,2))(x)
    x = keras.layers.Conv2D(8,(2,2),activation=activation,kernel_initializer=initializer,)(conv2)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(8,(2,2),activation=activation,kernel_initializer=initializer,)(x)
    x = keras.layers.BatchNormalization()(x)
    conv3 = keras.layers.MaxPool2D((2,2))(x)

    flat1 = keras.layers.Flatten()(conv1)
    flat2 = keras.layers.Flatten()(conv2)
    flat3 = keras.layers.Flatten()(conv3)
    x = keras.layers.concatenate([flat1,flat2,flat3])
    
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Flatten()(conv3)
    x = keras.layers.Dropout(0.9)(x)
    x = keras.layers.Dense(128,activation = activation,kernel_initializer=initializer)(x)
    x = keras.layers.Dropout(0.7)(x)
    x = keras.layers.Dense(64,activation = activation,kernel_initializer=initializer)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(32,activation = activation,kernel_initializer=initializer)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16,activation = activation,kernel_initializer=initializer)(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(8,activation = activation,kernel_initializer=initializer)(x)
    
    #x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(4,activation = activation,kernel_initializer=initializer)(x)
    #x = keras.layers.Dense(numOfDense1Units,activation = activation,kernel_initializer=initializer)(x)
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Dense(numOfDense2Units,activation = activation,kernel_initializer=initializer,)(x)#kernel_regularizer=regularization
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Dropout(dropoutRate)(x)
    #x = keras.layers.Dense(numOfDense3Units,activation = activation,kernel_initializer=initializer,)(x)
    #x = keras.layers.Dropout(dropoutRate)(x)
    #x = keras.layers.Dense(numOfDense3Units,activation = activation,kernel_initializer=initializer,)(x)
   
    out= keras.layers.Dense(2,name = "mainOut",activation = "softmax")(x)

    model = keras.Model(inputs = [input],outputs = out)
    model.compile(keras.optimizers.Nadam(lr),
                  loss = keras.losses.binary_crossentropy,
                  metrics = ["accuracy"],
                  )#.SGD(lr=lr,momentum=0.9,nesterov=True)
    model.summary()
    history = model.fit(DataX,LabelsY,
                        validation_split=0.5,#validation_data= (testData,testLabels)
                        batch_size=batchSize,
                        epochs=numEpochs,
                        shuffle=True,
                        )
    plotAccAndLoss(history,lr)

def summedSimple58valAcc(DataX,LabelsY,testData,testLabels,categorizeLabels,lr):
    from numpy import sum
    



    #transforming the data
    def summingData(data):
        out = []
        for c in data:
            out.append(asarray([sum(v) for _,v in enumerate(c)]))
        return transpose(out)
   
    def countingCracks(data,threchold,lenOfTheCrack):
        '''
        creating dataset of 4 numbers s.t. they are close to the real crack count
        '''
        
        trNumOfCracks = open(CONST.part2npTrainNumCracksPath)
        testNumOfCracks = open(CONST.part2npTestNumCracksPath)

        numCracksDataset = empty((shape(data)[0],CONST.numOfColours))
        
        def processPixel(x,y,indeciesUsed,currCrack,image,threchold, recDepth):
            '''
            returns whether there is a crack in the pixel   
            '''
            def getNeighbours(x,y):
                '''
                return the neigbours of the pixel
                '''
                neighbours = []
                if x!=0:
                    neighbours.append([x-1,y])
                    if y!=0:
                        neighbours.append([x-1,y-1])
                if x!=CONST.rescaleImageX//CONST.part2PredictionStrideX-1:
                    neighbours.append([x+1,y])
                    if y!=CONST.rescaleImageY//CONST.part2PredictionStrideY-1:
                        neighbours.append([x+1,y+1])
                if y!=0:
                    neighbours.append([x,y-1])
                    if x!=0:
                        neighbours.append([x-1,y-1])
                if y!=CONST.rescaleImageY//CONST.part2PredictionStrideY-1:
                    neighbours.append([x,y+1])
                    if x!=CONST.rescaleImageX//CONST.part2PredictionStrideX-1:
                        neighbours.append([x+1,y+1])
                return neighbours
                pass

            
            recDepth+=1
            if image[x][y]>threchold:
                if [x,y] not in indeciesUsed:            
                    indeciesUsed.append([x,y])
                    currCrack.append([x,y])
                    neighbours = getNeighbours(x,y)
                    for p in neighbours:
                        if recDepth<CONST.sTDAmaxRecursion:#controlling the maximum rec Depth
                            processPixel(p[0],p[1],indeciesUsed,currCrack,image,threchold,recDepth)
            recDepth-=1

            return currCrack,indeciesUsed
   
        for j,v in enumerate(data):
                
            crackCount=empty(CONST.numOfColours)

            for i,c in enumerate(v):
                cracks = []
                indeciesUsed = []
                for x in range(CONST.rescaleImageX//CONST.part2PredictionStrideX):
                    for y in range(CONST.rescaleImageY//CONST.part2PredictionStrideY):

                        crack,indeciesUsed = processPixel(x,y,indeciesUsed,[],c,threchold,0)
                        cracks.append(crack)
                #print(i," ",shape(empty))
                crackCount[i] = len([i for i in cracks if len(i)>lenOfTheCrack])
                if crackCount[i]>=8:
                    for k in c:
                       print(k)

            print(crackCount)
            numCracksDataset[j] = crackCount
        return numCracksDataset

    print(shape(DataX))
    #DataX = countingCracks(np.swapaxes(DataX,0,1),CONST.sTDAthrechold,CONST.sTDAcrackLength)
    DataX = summingData(DataX)

    #testData = countingCracks(np.swapaxes(testData,0,1),0.999,3)

    #labels to categotical data
    if categorizeLabels:
        LabelsY = keras.utils.to_categorical(labelsToClassification(LabelsY))
        testLabels = keras.utils.to_categorical(labelsToClassification(testLabels))

    #NN
    #parameters
    batchSize = 1000
    numEpochs = 300000


    #shuffling train data
    rng_state = np.random.get_state()
    np.random.shuffle(DataX)
    np.random.set_state(rng_state)
    np.random.shuffle(LabelsY)

    #shuffling test data
    rng_state = np.random.get_state()
    np.random.shuffle(testData)
    np.random.set_state(rng_state)
    np.random.shuffle(testLabels)


    regularizationRates = [0.01,0.01]
    numOfDense1Units = 32
    numOfDense2Units = 16
    numOfDense3Units = 8

    #initializer = keras.initializers.glorot_normal()
    #activation = "sigmoid"
    initializer = keras.initializers.he_normal()
    activation = "relu"

    regularization = keras.regularizers.L1L2(l1=regularizationRates[0],l2=regularizationRates[1])

    input = keras.layers.Input(shape=(shape(DataX[0])[0],),name="input")
    
    x= keras.layers.Dense(numOfDense1Units,activation = activation,kernel_initializer=initializer)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(numOfDense2Units,activation = activation,kernel_initializer=initializer,)(x)#kernel_regularizer=regularization
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(numOfDense3Units,activation = activation,kernel_initializer=initializer,)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(numOfDense3Units,activation = activation,kernel_initializer=initializer,)(x)
    x = keras.layers.Dense(numOfDense3Units,activation = activation,kernel_initializer=initializer,)(x)
    out= keras.layers.Dense(2,name = "mainOut",activation = "softmax")(x)
    
    model = keras.Model(inputs = [input],outputs = out)
    model.compile(keras.optimizers.Nadam(lr),loss = keras.losses.binary_crossentropy,metrics = ["accuracy"])#.SGD(lr=lr,momentum=0.9,nesterov=True)
    model.summary()
    history = model.fit(DataX,LabelsY,
                        validation_split=0.5,
                        #validation_data= (summingData(testData),testLabels),
                        batch_size=batchSize,
                        epochs=numEpochs,shuffle=False)#validation_data= (summingData(testData),testLabels)
    plotAccAndLoss(history,lr)

def summedSimpleGoodTrainingResults(DataX,LabelsY,testData,testLabels,categorizeLabels,lr):
    from numpy import sum

    #transforming the data
    def summingData(data):
        out = []
        for c in data:
            out.append(asarray([sum(v) for _,v in enumerate(c)]))
        return transpose(out)
    DataX = summingData(DataX)

    if categorizeLabels:
        LabelsY = labelsToClassification(LabelsY)
        testLabels = labelsToClassification(testLabels)

    #NN
    #parameters
    batchSize = 100
    numEpochs = 300000

    regularizationRates = [0.001,0.001]
    numOfDense1Units = 256
    numOfDense2Units = 512
    numOfDense3Units = 1014

    initializer = keras.initializers.glorot_normal()
    activation = "sigmoid"

    input = keras.layers.Input(shape=(shape(DataX[0])[0],),name="input")
    regularization = keras.regularizers.L1L2(l1=regularizationRates[0],l2=regularizationRates[1])
    x= keras.layers.Dense(numOfDense1Units,activation = activation,kernel_initializer=initializer)(input)
    x = keras.layers.Dense(numOfDense2Units,activation = activation,kernel_initializer=initializer)(x)
    x = keras.layers.Dense(numOfDense3Units,activation = activation,kernel_initializer=initializer)(x)
    out= keras.layers.Dense(1,name = "mainOut")(x)
    
    model = keras.Model(inputs = [input],outputs = out)
    model.compile(keras.optimizers.adam(lr),loss = keras.losses.mse,metrics = ["accuracy"])#.SGD(lr=lr,momentum=0.9,nesterov=True)
    model.summary()
    history = model.fit(DataX,LabelsY,validation_data= (summingData(testData),testLabels),batch_size=batchSize,epochs=numEpochs,shuffle=True)
    plotAccAndLoss(history,lr)

def summedOriginal(DataX,LabelsY,testDataAndLabels,lr = 1e-4):
   
    input = keras.layers.Input(shape=(4,),name="input")
    x= keras.layers.Dense(256,activation = "sigmoid",kernel_initializer="normal")(input)
    x = keras.layers.Dense(512,activation = "sigmoid",kernel_initializer="normal")(x)
    x = keras.layers.Dense(1024,activation = "sigmoid",kernel_initializer="normal")(x)
    x = keras.layers.Dense(1024,activation = "sigmoid",kernel_initializer="normal",kernel_regularizer = "l2")(x)
   
    out= keras.layers.Dense(2,name = "mainOut",activation="softmax" )(x)
    model = keras.Model(inputs = [input],outputs = out)
    model.compile(keras.optimizers.Adam(lr=1e-4),loss = keras.losses.categorical_crossentropy,metrics = ["accuracy"])
    model.summary()
    model.fit(DataX,LabelsY,validation_data= (summingData(testData),testLabels),batch_size=300,epochs=500000,shuffle=True)

def summed(DataX,LabelsY,testDataAndLabels,lr = 1e-4):
   
    print("lr= ",lr)

    input = keras.layers.Input(shape=(4,),name="input")
    x= keras.layers.Dense(256,activation = "sigmoid",kernel_initializer="normal")(input)
    x = keras.layers.Dense(512,activation = "sigmoid",kernel_initializer="normal")(x)
    x= keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024,activation = "sigmoid",kernel_initializer="normal",kernel_regularizer=keras.regularizers.l2(0.1))(x)
    x= keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024,activation = "sigmoid",kernel_initializer="normal")(x)

   
    out= keras.layers.Dense(2,name = "mainOut",activation="softmax" )(x)
    model = keras.Model(inputs = [input],outputs = out)
    model.compile(keras.optimizers.Adam(lr=lr),loss = keras.losses.categorical_crossentropy,metrics = ["accuracy"])
    model.summary()
    history = model.fit(DataX,LabelsY,validation_data=testDataAndLabels,batch_size=300,epochs=250000,shuffle=True)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(r"S:\convnet_smaller_images\ConvNet\painted_data\large_dataset\alpha_search\alpha_"+sys.argv[1]+".jpg")


def stepTwoSimpleModel(x,y):
    

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(4, 4),
                 activation='relu',
                 input_shape=(60,60,1),kernel_initializer="he_uniform"))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer="he_uniform"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='elu',kernel_initializer="he_uniform"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='softmax'))

   
    model.compile(keras.optimizers.adam(lr=float(sys.argv[1])),loss = keras.losses.categorical_crossentropy,metrics = ["accuracy"])
    model.summary()
    history = model.fit(x = x,y = y,batch_size=300,epochs=200,shuffle=True)
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
   
    plt.savefig(r"S:\convnet_smaller_images\ConvNet\painted_data\large_dataset\alpha_search\alpha_"+sys.argv[1]+".jpg")

def stepTwoNetwork(trData,trLabels,kernel,numEpochs,testDataLabels):

    print(numpy.shape(trLabels))

    convLayerDim = 32
    dense1Dim = 128
    dense2Dim = 256
    convDence =  128


    yellowInput = keras.layers.Input(shape = (60,60,1),name = "yellowInput")
    yellowConv = keras.layers.Conv2D(convLayerDim,kernel_size=kernel,activation = "sigmoid",kernel_initializer="normal")(yellowInput) 
    yellowConv2 = keras.layers.Conv2D(convLayerDim*2,kernel_size=kernel,activation = "sigmoid",kernel_initializer="normal")(yellowConv)
    yellowPool = keras.layers.AveragePooling2D()(yellowConv2)
    Ydrop  = keras.layers.Dropout(0.25)(yellowPool)

    redInput = keras.layers.Input(shape = (60,60,1),name = "redInput")
    redConv = keras.layers.Conv2D(convLayerDim,kernel_size=kernel,activation = "sigmoid",kernel_initializer="normal")(redInput)
    redConv2 = keras.layers.Conv2D(convLayerDim*2,kernel_size=kernel,activation = "sigmoid",kernel_initializer="normal")(redConv)
    redPool = keras.layers.AveragePooling2D()(redConv2) 
    Rdrop = keras.layers.Dropout(0.25)(redPool)

    blueInput = keras.layers.Input(shape = (60,60,1),name = "blueInput")
    blueConv = keras.layers.Conv2D(convLayerDim,kernel_size=kernel,activation = "sigmoid",kernel_initializer="normal")(blueInput)
    blueConv2 = keras.layers.Conv2D(convLayerDim*2,kernel_size=kernel,activation = "sigmoid",kernel_initializer="normal")(blueConv)
    bluePool = keras.layers.AveragePooling2D()(blueConv2)
    Bdrop = keras.layers.Dropout(0.25)(bluePool)

    greenInput = keras.layers.Input(shape = (60,60,1),name = "greenInput")
    greenConv = keras.layers.Conv2D(convLayerDim,kernel_size=kernel,activation = "sigmoid",kernel_initializer="normal")(greenInput)
    greenConv2 = keras.layers.Conv2D(convLayerDim*2,kernel_size=kernel,activation = "sigmoid",kernel_initializer="normal")(greenConv)
    greenPool = keras.layers.AveragePooling2D()(greenConv2) 
    Gdrop = keras.layers.Dropout(0.25)(greenPool)

    x = keras.layers.concatenate([Ydrop,Rdrop,Bdrop,Gdrop])
    x = keras.layers.Flatten()(x)
    
    x = keras.layers.Dense(dense1Dim,activation = "sigmoid",kernel_initializer="normal")(x)
    x = keras.layers.LeakyReLU(0.3)(x)
    x = keras.layers.Dropout(0.5)(x)
   
   
    output = keras.layers.Dense(2,name = "mainOut",activation='softmax')(x)
    model  = keras.Model(inputs = [yellowInput,redInput,blueInput,greenInput],outputs = output)
    optimizer = keras.optimizers.adam(lr = 1e-5)#float(sys.argv[1]))#SGD(lr= 1e-5)
    loss = keras.losses.categorical_crossentropy
    model.compile(loss = loss,optimizer=optimizer,metrics = ["accuracy"])
    model.summary()
    history = model.fit(epochs= numEpochs,batch_size=50,x= trData,y= trLabels,validation_data=testDataLabels,shuffle = True)
    
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
   
    #plt.savefig(r"S:\convnet_smaller_images\ConvNet\painted_data\large_dataset\alpha_search\alpha_"+sys.argv[1]+".jpg")
    
    pass

