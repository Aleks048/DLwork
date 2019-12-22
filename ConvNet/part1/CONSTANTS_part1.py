import CONSTANTS as CONST
'''
constants used to train the model for part 1 are stored here
'''

'''parameters general'''
part1Debuging = CONST.part1Debuging

'''parameters related to cross-validation'''
useCrossValidation = False
numKFolds = 5


'''parameters used when either train the model to have 4 classes(by the number of colors) or to detect whther the image has a crack or not'''
trainColorNum = -1
numClasses = 4 if trainColorNum == -1 else 2

numYellowCrack = 1 if trainColorNum==0 else (0 if trainColorNum==-1 else 0)
numRedCrack = 1 if trainColorNum==1 else (1 if trainColorNum==-1 else 0)
numBlueCrack  = 1 if trainColorNum==2 else (2 if trainColorNum==-1 else 0)
numGreenCrack = 1 if trainColorNum==2 else (2 if trainColorNum==-1 else 0)
numNoCrack = 1 if trainColorNum==3 else (3 if trainColorNum==-1 else 0)



'''weights for each class if the weights used in the model'''
if numClasses==4:
    part1weightDict = [1/12,1/3,1/7,1/50]#the weights are obrained from the data distribution and can be anytheng// they  are Yellow,Red,Blue,Green
else:
    part1weightDict = [1,1]


'''parameters of the NN model for part 1'''
import keras
import part1networks
#input dimensions
input_XYdim = CONST.part1StrideX,CONST.part1StrideY
imput_num_channels = CONST.part1NumChannels
part1StrideX = CONST.part1StrideX
part1StrideY = CONST.part1StrideY
part1NumChannels = CONST.part1NumChannels
generator_in_shape =  (part1StrideX,part1StrideY,part1NumChannels)
#optimizer
alpha = 1.3/pow(5,7)# can use 6 instead of 7
beta_1 = 0.9
beta_2 = 0.99
decay = (1/100)*alpha
epsilon = 10e-8
amsGrad = True
modelKernelInitializer="he_uniform"
optimizer=keras.optimizers.Adam(lr=alpha,beta_1=beta_1,beta_2=beta_2,decay=decay,epsilon = epsilon,amsgrad=amsGrad)
#loss
loss = keras.losses.categorical_crossentropy
#model
model = part1networks.leNetExp(input_shape=generator_in_shape,num_classes=numClasses)


'''parameters related to training the model for part 1'''
batch_size = 100
nEpochs = 1
data_generator_params = {'dim': input_XYdim,
                         'batch_size': batch_size,
                         'n_classes':numClasses,
                         'n_channels': part1NumChannels,
                         'shuffle': True,
                         'useConv': True
                         }
training_parameters={"epochs":nEpochs,
                    "use_multiprocessing":False,
                     "workers":1,
                     "max_queue_size":1,
                     "verbose":1}


'''parameters related to pretrained network if used'''
part1usePretrainedNetwork = False#use pretrained network or not
#loading model
if part1usePretrainedNetwork:
    from keras.applications import VGG19
    part1pathToDataGeneratedByPretrainedNet = r"..\complete dataset\colored_images_and_np_arrays_from_them\dataNamesAndLabelsArrays\generated_by_vgg19"#should we generate if first??
    pretrainedNetPart1 = VGG19(weights='imagenet',#loading the net
                      include_top=False,
                      input_shape=(32, 32, 3))#the parameters should be defined somewhere??
#training
batch_size_pretrainedNet = 30
input_XYdim_pretrainedNet = (512,0)
input_num_channels_pretrainedNet = 0
generator_in_shape_pretrainedNet = [input_XYdim_pretrainedNet[0]]#could use a separate parameter for this one
model_pretrainedNet = part1networks.leNetExp(input_shape=generator_in_shape_pretrainedNet,num_classes=numClasses)
data_generator_params_pretrainedNet = {'dim': input_XYdim_pretrainedNet,
                 'batch_size': batch_size_pretrainedNet,
                 'n_classes':numClasses,
                 'n_channels': input_num_channels_pretrainedNet,
                 'shuffle': True,
                 'useConv': False
                 }


'''paths'''
part1trainedModelSavePath = r"..\complete dataset\colored_images_and_np_arrays_from_them\trainedModelsPart1\part1trainedModel.h5"
