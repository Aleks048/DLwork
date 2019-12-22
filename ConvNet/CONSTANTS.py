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


'''
part 2
'''

createThePart1PredictedDataset = False
useFourMachinesFromPart1 = False

#paths to models used in part 2 generated from previous part
yellowModelPath =r"..\complete dataset\machines_trained_on_colored_images\old\trained_before\yellow.h5"
redModelPath = r"..\complete dataset\machines_trained_on_colored_images\old\trained_before\red.h5"
blueModelPath =r"..\complete dataset\machines_trained_on_colored_images\old\trained_before\blue.h5"
greenModelPath =r"..\complete dataset\machines_trained_on_colored_images\old\trained_before\green.h5"


# path to folder that contains large dataset of images with dris
useDataset1 = False

#path to the spreadsheet with dris\
if useDataset1:
    driSpreadsheetPath = r"..\complete dataset\images_with_dri\DRI_for_the_images.xlsx"
else:
    driSpreadsheetPath = r"..\complete dataset\large_images_with_dri_part2_Naiara\DRI NRT AAA-AAH #1 - 06.10.19.xlsx"

if useDataset1:
    largeDatasetPath  = r"..\complete dataset\images_with_dri"
else:
    largeDatasetPath  = r"..\complete dataset\large_images_with_dri_part2_Naiara"

#path to json dataset predicted by models from part 1
jsonDatasetPath = r"..\complete dataset\images_with_dri\predictions_and_DRIs.json"
jsonWithNumCracksPath = r"..\complete dataset\images_with_dri\predictions_and_DRIs_andNumOfCracks.json"

#path to numpy array for the training and test data
part2npTrainDatasetPath = r"..\complete dataset\images_with_dri\trainTestDataLabels\trainingData.npy"
part2npTrainLabelsPath = r"..\complete dataset\images_with_dri\trainTestDataLabels\trLabels.npy"
part2npTrainNumCracksPath = r"..\complete dataset\images_with_dri\trainTestDataLabels\trNumCracks.npy"
part2npTestDatasetPath = r"..\complete dataset\images_with_dri\trainTestDataLabels\testData.npy"
part2npTestLabelsPath = r"..\complete dataset\images_with_dri\trainTestDataLabels\testLabels.npy"
part2npTestNumCracksPath = r"..\complete dataset\images_with_dri\trainTestDataLabels\testNumCracks.npy"






#part 2 strides used for prediction when the step 2 dataset is created
part2PredictionStrideX = 30
part2PredictionStrideY = 30

#large dataset training testing split
largeDatasetTrTestingSplit = 0.8

#part 2 NN acc and loss save
part2pathAccSave = r"..\painted_data\large_dataset\alpha_search\current\acc_"
part2pathLossSave =  r"..\painted_data\large_dataset\alpha_search\current\loss_"



#simple TDA parameters//if the fancy counting of the cracks for part 2 is used
sTDAcrackLength = 200
sTDAthrechold = 0.99
sTDAmaxRecursion = 60




'''
part 1
'''

#common
#used in preprocessing and training part 1 model
#it is also  the dimension of the cutted image in the preprocessing phase
part1StrideX = 90
part1StrideY = 90
part1NumChannels = 3

#preprocessing
folderNames = ["2","3","4","5"]#why 1 is excluded????
listOfColorsUsed =["yellow","red","blue","green","noCracks"]
augTypes = ["0rot","90rot","180rot","270rot","0ref","90ref","180ref","270ref"]#0rot should allways be first in the list

part1Debuging = True



#
#
#loading the dataset and preparing for training
#
#



part1ArraysOFDataNamesAndLabelsPath = r"..\complete dataset\colored_images_and_np_arrays_from_them\dataNamesAndLabelsArrays"

part1trTestSplit = 0.8 #if not useCrossValidation else 1/numKFolds





