from keras.models import load_model
from os import walk
from os.path import join
from json import dump as jdump
import json
from numpy import asarray,expand_dims,swapaxes,shape,squeeze,empty
import numpy as np
from PIL.Image import open as imgOpen
import CONSTANTS as CONST
from os import fsencode,listdir,fsdecode
from xlrd import open_workbook

'''
tested everything here worked as expected
'''


def fileNameToConsistentFormat(filename):
    out = filename.replace("O","0")
    out = out.replace(" ","")
    out = out.replace(".","")
    out = out.replace("(camera305)","")
    out = out.replace("2(camera305)","")
    out = out.replace("(","")
    out = out.replace(")","")
    out = out.replace("01","1")
    out = out.replace("02","2")
    out = out.replace("03","3")
    out = out.replace("04","4")
    out = out.replace("05","5")
    out = out.replace("06","6")
    out = out.replace("07","7")
    out = out.replace("08","8")
    out = out.replace("09","9")

    return out

def getImageDri(imageName:str,book):
    '''
    finds the dri for the image in the sheet that is stored in the same folder as the image
    
    imageName without extension
    
    the book must have the name of the spreadsheet as first 3 letters of the spreadsheet


    returns an images dri
    '''
    numLettersInTheSheetName = 3 if CONST.useDataset1 else 7#Plus 1

    try:
        sheet = book.sheet_by_name(imageName[0:numLettersInTheSheetName])
    except:
        if CONST.useDataset1:
            raise Exception("Could not find the the spreadsheet. Name:"+imageName[0:3])
        else:
            raise Exception("Could not find the the spreadsheet. Name:"+imageName[0:7])
    for row in range(sheet.nrows):
        updatedimgNameFromSpreadSh = fileNameToConsistentFormat(sheet.cell_value(row,1))
        if CONST.useDataset1:
            updatedimgNameFromFile = fileNameToConsistentFormat(imageName[numLettersInTheSheetName+1:])
        else:
            updatedimgNameFromFile = "R1"+fileNameToConsistentFormat(imageName[numLettersInTheSheetName:])

        if (sheet.cell(row,0).value==imageName[:numLettersInTheSheetName]) and (updatedimgNameFromSpreadSh==updatedimgNameFromFile):
            return sheet.cell(row,2).value
    raise Exception("Could not find the image in the spreadsheets. Name:"+imageName)

def collectPredictionsForTheImage(model,imagePath:str,stepX:int,stepY:int):
    '''
    runs through the 30x30 grids of the image and predicts for each of them
     
    name of the ,model included in the pathand the extension / name of the image included in the path and the extension
    
    returns an array of predictions of the image (percentage for each class)
    
    '''
    #model = keras.models.load_model(modelPath)
    im = asarray((imgOpen(imagePath)).convert("RGB").resize((CONST.rescaleImageX,CONST.rescaleImageY)))
    
    
    x=0
    y=0
    if CONST.useDataset1:
        outPredictions = []#the predictions arraays will be stored here
    else:
        outPredictions = empty((CONST.numOfColours,CONST.rescaleImageX//stepX,CONST.rescaleImageY//stepY))

    while x<CONST.rescaleImageX:
        while y<CONST.rescaleImageY:
            predicted = model.predict(expand_dims(im[x:x+stepX,y:y+stepY],axis=0))
            if CONST.useFourMachinesFromPart1:
                outPredictions.append(predicted[0].tolist())
            else:
                for c,p in enumerate(predicted[0]):
                    outPredictions[c][x//30][y//30] = p
            y+=stepY
        x+=stepX
        y=0

    return outPredictions




def processTheImages(models,dictSavePath:str,rootFolderPath:str,driSpreadsheetPath:str,stepX:int,stepY:int):
    '''
    
    runs collectPredictionsForImages and getImageDRI for All the Images in the rootFolderPath recursively

    dictSavePath path +name+extension/saves as json


    returns dict key = imageName , value  = arrayOfPredictions and dri of the image
    '''
    out = {}

    book = open_workbook(driSpreadsheetPath)

    for root,_,files in walk(rootFolderPath):
        for f in files:
            print("hi")
            if ".jpg" in f:
                print(f)#logging the files passed
                
                if CONST.useFourMachinesFromPart1:
                    predictionsYellow = collectPredictionsForTheImage(models["Yellow"],join(root,f),stepX,stepY)
                    predictionsRed = collectPredictionsForTheImage(models["Red"],join(root,f),stepX,stepY)
                    predictionsBlue = collectPredictionsForTheImage(models["Blue"],join(root,f),stepX,stepY)
                    predictionsGreen = collectPredictionsForTheImage(models["Green"],join(root,f),stepX,stepY)
                else:
                    predictions = collectPredictionsForTheImage(models["model"],join(root,f),stepX,stepY)
                    print(shape(predictions))
                    predictionsYellow = np.ndarray.flatten(predictions[0]).tolist()
                    
                    predictionsRed = np.ndarray.flatten(predictions[1]).tolist()
                    predictionsBlue = np.ndarray.flatten(predictions[2]).tolist()
                    predictionsGreen = np.ndarray.flatten(predictions[3]).tolist()

                dri = getImageDri(f[:-4],book)
                out[join(root,f)] = [dri,[predictionsYellow,predictionsRed,predictionsBlue,predictionsGreen]]
    
    with open(dictSavePath,'w') as dp:#wite to json file
        jdump(out,dp)
    return out


def creatingPredictedDataset():
   '''
   for creating the predicted dataset

   requirements : the JSON file must already exist in the location
   '''

   models = {}
   if CONST.useFourMachinesFromPart1:
       models["Yellow"]= load_model(CONST.yellowModelPath)
       models["Red"]= load_model(CONST.redModelPath)
       models["Blue"]= load_model(CONST.blueModelPath)
       models["Green"]= load_model(CONST.greenModelPath)
   else:
       models["model"] = load_model(CONST.part1trainedModelSavePath)

   processTheImages(models,CONST.jsonDatasetPath,CONST.largeDatasetPath,CONST.driSpreadsheetPath,CONST.part2PredictionStrideX,CONST.part2PredictionStrideY)

   return models


if __name__ =="__main__":
    import os
    print(os.getcwd())
    creatingPredictedDataset()
else:
    print(__name__)