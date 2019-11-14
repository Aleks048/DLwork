'''
all global constants are kept here

'''

'''
CCA - Crack closed in coarse aggregate - Yellow
OCA - Crack open in coarse aggregate - Blue
OCAG - Crack open in coarse aggregate with gel - Green
CCP - Crack in cement paste - Red
'''





rescaleImageX = 1800
rescaleImageY = 1800
numOfColours = 4



#part 2


createThePart1PredictedDataset = False
useFourMachinesFromPart1 = False

#paths to models used in part 2 generated from previous part
yellowModelPath =r"S:\convnet_smaller_images\ConvNet\complete dataset\machines_trained_on_colored_images\old\trained_before\yellow.h5"
redModelPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\machines_trained_on_colored_images\old\trained_before\red.h5"
blueModelPath =r"S:\convnet_smaller_images\ConvNet\complete dataset\machines_trained_on_colored_images\old\trained_before\blue.h5"
greenModelPath =r"S:\convnet_smaller_images\ConvNet\complete dataset\machines_trained_on_colored_images\old\trained_before\green.h5"


# path to folder that contains large dataset of images with dris
useDataset1 = False

#path to the spreadsheet with dris\
if useDataset1:
    driSpreadsheetPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri\DRI_for_the_images.xlsx"
else:
    driSpreadsheetPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\large_images_with_dri_part2_Naiara\DRI NRT AAA-AAH #1 - 06.10.19.xlsx"

if useDataset1:
    largeDatasetPath  = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri"
else:
    largeDatasetPath  = r"S:\convnet_smaller_images\ConvNet\complete dataset\large_images_with_dri_part2_Naiara"

#path to json dataset predicted by models from part 1
jsonDatasetPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri\predictions_and_DRIs.json"
jsonWithNumCracksPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri\predictions_and_DRIs_andNumOfCracks.json"

#path to numpy array for the training and test data
part2npTrainDatasetPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri\trainTestDataLabels\trainingData.npy"
part2npTrainLabelsPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri\trainTestDataLabels\trLabels.npy"
part2npTrainNumCracksPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri\trainTestDataLabels\trNumCracks.npy"
part2npTestDatasetPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri\trainTestDataLabels\testData.npy"
part2npTestLabelsPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri\trainTestDataLabels\testLabels.npy"
part2npTestNumCracksPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\images_with_dri\trainTestDataLabels\testNumCracks.npy"






#part 2 strides used for prediction when the step 2 dataset is created
part2PredictionStrideX = 30
part2PredictionStrideY = 30

#large dataset training testing split
largeDatasetTrTestingSplit = 0.8

#part 2 NN acc and loss save
part2pathAccSave = r"S:\convnet_smaller_images\ConvNet\painted_data\large_dataset\alpha_search\current\acc_"
part2pathLossSave =  r"S:\convnet_smaller_images\ConvNet\painted_data\large_dataset\alpha_search\current\loss_"



#simple TDA parameters//if the fancy counting of the cracks for part 2 is used
sTDAcrackLength = 200
sTDAthrechold = 0.99
sTDAmaxRecursion = 60




'''
part 1
'''
#
#
#cut the images// see if there is a crack
#

#common
part1StrideX = 90
part1StrideY = 90
part1NumChannels = 3

folderNames = ["2","3","4","5"]
listOfColorsUsed =["yellow","red","blue","green","noCracks"]
augTypes = ["0rot","90rot","180rot","270rot","0ref","90ref","180ref","270ref"]#0rot should allways be first in the list

part1Debuging = True

#creatingTheDataset
pathToColoredImagesRootFolder = r"S:\convnet_smaller_images\ConvNet\complete dataset\colored_images_and_np_arrays_from_them"

useMiltithreading = False
coloredImagesAreCutAndNumpyArraysCreated = True
theNoCrackDatasetIsCreated = True
augmentedDataCreated = True
theDataSetFromColoredCracksIsGenerated = True
imagesCutArraysCreated = True


rejection_colour_parameters = {"Yellow":[range(245,256),range(245,256),range(97,107)],
                               "Yellow2":[range(200,256),range(200,256),range(0,45)],
                               "Red":[range(200,256),range(0,70),range(0,70)],
                               "Red2":[range(200,256),range(0,40),range(0,40)],
                               "Blue":[range(97,107),range(97,107),range(245,256)],
                               "Blue2":[range(0,40),range(0,40),range(200,256)],
                               "Green":[range(0,120),range(180,256),range(0,120)],
                               "Green2":[range(0,40),range(200,256),range(0,40)]}


coloredImageBoundary = 7#used to avoid cracks that appear in close to the boundary of the cutted image

percentageOfTheSmallImageTakenByTheCrack = 0.04/3.5
percentageOfTheSmallImageTakenByNOnPrimaryCrack = 0.001

colored_and_notColored_img_separator = "COLOR LINES"

#
#
#loading the dataset and preparing for training
#
#

trainColorNum = -1
numClasses = 4 if trainColorNum == -1 else 2

numYellowCrack = 1 if trainColorNum==0 else (0 if trainColorNum==-1 else 0)
numRedCrack = 1 if trainColorNum==1 else (1 if trainColorNum==-1 else 0)
numBlueCrack  = 1 if trainColorNum==2 else (2 if trainColorNum==-1 else 0)
numGreenCrack = 1 if trainColorNum==2 else (2 if trainColorNum==-1 else 0)
numNoCrack = 1 if trainColorNum==3 else (3 if trainColorNum==-1 else 0)

if numClasses==4:
    part1weightDict = [1/12,1/3,1/7,1/50]
else:
    part1weightDict = [1,1]

part1ArraysOFDataNamesAndLabelsPath = r"S:\convnet_smaller_images\ConvNet\complete dataset\colored_images_and_np_arrays_from_them\dataNamesAndLabelsArrays"

useCrossValidation = False
numKFolds = 5

part1trTestSplit = 0.8 if not useCrossValidation else 1/numKFolds


part1usePretrainedNetwork = False
if part1usePretrainedNetwork:
    from keras.applications import VGG19
    part1pathToDataGeneratedByPretrainedNet = r"S:\convnet_smaller_images\ConvNet\complete dataset\colored_images_and_np_arrays_from_them\dataNamesAndLabelsArrays\generated_by_vgg19"
    pretrainedNetPart1 = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(32, 32, 3))


part1trainedModelSavePath = r"S:\convnet_smaller_images\ConvNet\complete dataset\colored_images_and_np_arrays_from_them\trainedModelsPart1\part1trainedModel.h5"