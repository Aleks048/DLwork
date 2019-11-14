import CONSTANTS
from xlrd import open_workbook

'''
create dataset of predictions
'''
from create_dataset_of_predictions import *

''' testing get images dri/PASSED'''

#pathTestDriSpreadSheet = r"S:\convnet_smaller_images\ConvNet\tests\test_DRI_for_the_images.xlsx"#book has only AAB spreadsheet

#book = open_workbook(CONSTANTS.driSpreadsheetPath)
#getImageDri("NRT AAA F (1)",book)

#print(getImageDri('AAB A02 R0 A',book))#correct
#getImageDri('AAB e15',book)#wrong image name
#getImageDri('ADB e15',book)#wrong spreadsheetName

'''testing processTheImages/PASSED'''

#testDatasetPath = r"S:\convnet_smaller_images\ConvNet\tests\test_large_dataset\AAB R0 A - J+D 45  - 19 -0.05%"
#testDatasetJSON = r"S:\convnet_smaller_images\ConvNet\tests\test_large_dataset\path_dri_predictions.json"
#models = {}
    
#models["Yellow"]= load_model(CONST.yellowModelPath)
#models["Red"]= load_model(CONST.redModelPath)
#models["Blue"]= load_model(CONST.blueModelPath)
#models["Green"]= load_model(CONST.greenModelPath)
#models["model"] = load_model(CONST.part1trainedModelSavePath)
#processTheImages(models,testDatasetJSON,testDatasetPath,CONST.driSpreadsheetPath,CONST.part2PredictionStrideX,CONST.part2PredictionStrideY)#check if the function works as expected

