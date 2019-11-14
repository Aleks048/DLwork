'''
here we go through the pained images and mark the positions of pained cracks and create the numpy arrays
'''


import random
import datetime
from MyImage import MyImage
import os

import numpy
from PIL import Image
from difflib import ndiff

import _thread,threading

from multiprocessing.pool import ThreadPool

import CONSTANTS as CONST

from random import shuffle
import json

import pickle

from random import sample


#used for creating dataSet
def countThePixelsOfColours(arrayPaint,arrayNoPaint,x,y):
    '''
    given the image and starting coordinates an image of the CONST.part1strideX by CONST.part1strideY is cut
    then the number of pixels of certain colours is counted

    used by cutTheimageCreateArray and
    '''

    #2 colors since sometimes the colours that were used were different
    yellow=0
    yellow2=0
    blue=0
    blue2=0
    red = 0
    red2 = 0
    green = 0
    green2 = 0

    cutImage = arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]

    boundary =CONST.coloredImageBoundary#to avoid cracks appear on the boundaries
    for i in range(boundary,CONST.part1StrideX-boundary):
        for j in range (boundary,CONST.part1StrideY-boundary):
                  
            if cutImage[i][j][0] in CONST.rejection_colour_parameters["Yellow"][0] and cutImage[i][j][1] in CONST.rejection_colour_parameters["Yellow"][1] and cutImage[i][j][2] in CONST.rejection_colour_parameters["Yellow"][2] :
                yellow+=1
            if cutImage[i][j][0] in CONST.rejection_colour_parameters["Yellow2"][0] and cutImage[i][j][1] in CONST.rejection_colour_parameters["Yellow2"][1] and cutImage[i][j][2] in CONST.rejection_colour_parameters["Yellow2"][2] :
                yellow2+=1
            if cutImage[i][j][0] in CONST.rejection_colour_parameters["Red"][0] and cutImage[i][j][1] in CONST.rejection_colour_parameters["Red"][1] and cutImage[i][j][2] in CONST.rejection_colour_parameters["Red"][2]:
                red+=1
            if cutImage[i][j][0] in CONST.rejection_colour_parameters["Red2"][0] and cutImage[i][j][1] in CONST.rejection_colour_parameters["Red2"][1] and cutImage[i][j][2] in CONST.rejection_colour_parameters["Red2"][2]:
                red2+=1
            if cutImage[i][j][0] in CONST.rejection_colour_parameters["Blue"][0] and cutImage[i][j][1] in CONST.rejection_colour_parameters["Blue"][1] and cutImage[i][j][2]in CONST.rejection_colour_parameters["Blue"][2] :
                blue+=1
            if cutImage[i][j][0] in CONST.rejection_colour_parameters["Blue2"][0] and cutImage[i][j][1] in CONST.rejection_colour_parameters["Blue2"][1] and cutImage[i][j][2]in CONST.rejection_colour_parameters["Blue2"][2] :
                blue2+=1
            if cutImage[i][j][0] in CONST.rejection_colour_parameters["Green"][0] and cutImage[i][j][1] in CONST.rejection_colour_parameters["Green"][1] and cutImage[i][j][2]in CONST.rejection_colour_parameters["Green"][2] :
                green+=1
            if cutImage[i][j][0] in CONST.rejection_colour_parameters["Green2"][0] and cutImage[i][j][1] in CONST.rejection_colour_parameters["Green2"][1] and cutImage[i][j][2]in CONST.rejection_colour_parameters["Green2"][2] :
                green2+=1
        
    return yellow,yellow2,red,red2,blue,blue2,green,green2   

def saveAugmentedImages(name:str,pathToOrigImg:str,savePath:str):
    '''
    used for creating the augmented data
    '''
    rotated0 = numpy.load(pathToOrigImg)
                
    rotated90 = numpy.rot90(rotated0,axes=(-3,-2))
    rotated180 = numpy.rot90(rotated90,axes=(-3,-2))     
    rotated270 = numpy.rot90(rotated180,axes=(-3,-2))
            
    #reflections arrays
    reflection0 = numpy.fliplr(rotated0)
    reflection90 = numpy.fliplr(rotated90)
    reflection180 = numpy.fliplr(rotated180)
    reflection270 = numpy.fliplr(rotated270)
                
           
            
    #saving
    numpy.save(savePath+r"\90rot"+"\\"+name+"_90rot.npy",rotated90)



    numpy.save(savePath+r"\180rot"+"\\"+name+"_180rot.npy",rotated180)
    numpy.save(savePath+r"\270rot"+"\\"+name+"_270rot.npy",rotated270)

    numpy.save(savePath+r"\0ref"+"\\"+name+"_0ref.npy",reflection0)
    numpy.save(savePath+r"\90ref"+"\\"+name+"_90ref.npy",reflection90)
    numpy.save(savePath+r"\180ref"+"\\"+name+"_180ref.npy",reflection180)
    numpy.save(savePath+r"\270ref"+"\\"+name+"_270ref.npy",reflection270)

def CreateImagesWithoutCracks(pathPaint:str,pathNoPaint:str,numOfImagesToObtain:int,savePath:str,imgName:str):
    imgFull = Image.open(pathPaint,mode="r").convert("RGB")
    imgResized = imgFull.resize((CONST.rescaleImageX,CONST.rescaleImageY))
    arrayPaint = numpy.asarray(imgResized)
    imgFullNoPaint = Image.open(pathNoPaint)
    imgFullNoPaintResized =imgFullNoPaint.resize((CONST.rescaleImageX,CONST.rescaleImageY)) 
    arrayNoPaint = numpy.asarray(imgFullNoPaintResized)
    
    
    countNoCrack = 0
    x=0
    y=0
    while y in range(0,CONST.rescaleImageY-CONST.part1StrideY+1):
        print(y)

        while x in range(0,CONST.rescaleImageY-CONST.part1StrideX+1):
            yellow,yellow2,red,red2,blue,blue2,green,green2 = countThePixelsOfColours(arrayPaint,arrayNoPaint,x,y)
            if  (
                (green<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                (green2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                (red<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                (red2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                (blue<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                (blue2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                (yellow<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                (yellow2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) 
                ):
                numpy.save(savePath+r"\numpy_arrays"+r"\noCracks\0rot"+"\\"+imgName+"noCracks_"+str(countNoCrack+1)+".npy",arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX])
                Image.fromarray(arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\noCracks\0rot"+"\\"+imgName+"_noCracks_"+str(countNoCrack+1)+".jpg","JPEG")
                Image.fromarray(arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\noCracks\0rot"+"\\"+imgName+"_noCracks_"+str(countNoCrack+1)+"_noPaint.jpg","JPEG")
                countNoCrack+=1
            x+=CONST.part1StrideX
        
        x=0
        y+=CONST.part1StrideY
    '''
    #used if we need sampling
    #count = numOfImagesToObtain
    #used =[]
    #while count!=0:
    #    notUsed=True
    #    x=random.randint(0,originaImageSizeX-imageGridX)
    #    y=random.randint(0,originaImageSizeY-imageGridY)
    #    for pair in used:
    #        if x in range(pair[0]-5,pair[0]+5) and y in range(pair[1]-5,pair[1]+5):
    #            notUsed=False
    #            break
    #    if notUsed:
    #        yellow,yellow2,red,red2,blue,blue2,green,green2 = countThePixelsOfColours(arrayPaint,arrayNoPaint,x,y)
    #        if (yellow<0.0001*imageGridX*imageGridY) and (red<0.0001*imageGridX*imageGridY) and (blue<0.0001*imageGridX*imageGridY) and (green<0.0001*imageGridX*imageGridY):
    #            used.append([x,y])
    #            numpy.save(savePath+r"\noCracks\0rot"+"\\"+imgName+"_NoCracks_"+str(count)+".npy",arrayNoPaint[x:x+imageGridX,y:y+imageGridY])
    #            count-=1
    '''

def imageToNumpyArrays(pathPaint:str,pathNoPaint:str,savePath:str,imgName:str,):
    '''
    given a large image 1800 by 1800
    it is cut into smaller images/images with cracks identified and saved accordingly
    '''

    def cutTheimagesCreateArrays(arrayPaint,arrayNoPaint):

        countOfYellowImages=0
        countOfRedImages=0
        countOfBlueImages = 0
        countOfGreenImages = 0
        x=0
        y=0
        while y in range(0,CONST.rescaleImageY-CONST.part1StrideY+1):
            print(y)

            while x in range(0,CONST.rescaleImageY-CONST.part1StrideX+1):
                #count the pixels in the small image
                yellow,yellow2,red,red2,blue,blue2,green,green2 = countThePixelsOfColours(arrayPaint,arrayNoPaint,x,y)
     

                #check if there is a crack/there is enough pixels of certain colour and save the array if necessary
                if  (
                        (green<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (green2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (blue<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (blue2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (yellow>CONST.percentageOfTheSmallImageTakenByTheCrack*CONST.part1StrideX*CONST.part1StrideY)):#yellow
               
                    print("yellow",yellow)
                    numpy.save(savePath+"numpy_arrays"+r"\yellow\0rot"+"\\"+imgName+"Yellow_"+str(countOfYellowImages+1)+"_"+str(yellow)+".npy",arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX])
                    Image.fromarray(arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\yellow\0rot"+"\\"+imgName+"_Yellow_"+str(countOfYellowImages+1)+"_"+str(yellow)+".jpg","JPEG")
                    Image.fromarray(arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\yellow\0rot"+"\\"+imgName+"_Yellow_"+str(countOfYellowImages+1)+"_"+str(yellow)+"_noPaint.jpg","JPEG")
                    countOfYellowImages+=1
                elif (
                        (green<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (green2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (blue<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (blue2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (yellow2>CONST.percentageOfTheSmallImageTakenByTheCrack*CONST.part1StrideX*CONST.part1StrideY)):#yellow
               
                    print("yellow",yellow2)
                    numpy.save(savePath+"numpy_arrays"+r"\yellow\0rot"+"\\"+imgName+"Yellow2_"+str(countOfYellowImages+1)+"_"+str(yellow2)+".npy",arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX])
                    Image.fromarray(arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\yellow\0rot"+"\\"+imgName+"_Yellow2_"+str(countOfYellowImages+1)+"_"+str(yellow2)+".jpg","JPEG")
                    Image.fromarray(arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\yellow\0rot"+"\\"+imgName+"_Yellow2_"+str(countOfYellowImages+1)+"_"+str(yellow2)+"_noPaint.jpg","JPEG")
                    countOfYellowImages+=1
                elif (
                        (green<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and
                        (green2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and
                        (yellow<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and
                        (yellow2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and
                        (blue<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (blue2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red2>CONST.percentageOfTheSmallImageTakenByTheCrack*CONST.part1StrideX*CONST.part1StrideY)):#red
               
                    print("red:",red2)
                    numpy.save(savePath+"numpy_arrays"+r"\red\0rot"+"\\"+imgName+"Red2_"+str(countOfRedImages+1)+"_"+str(red2)+".npy",arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX])
                    Image.fromarray(arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\red\0rot"+"\\"+imgName+"_Red2_"+str(countOfRedImages+1)+"_"+str(red2)+".jpg","JPEG")
                    Image.fromarray(arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\red\0rot"+"\\"+imgName+"_Red2_"+str(countOfRedImages+1)+"_"+str(red2)+"_noPaint.jpg","JPEG")
                    countOfRedImages+=1
                elif (
                        (green<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and
                        (green2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and
                        (yellow<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and
                        (yellow2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and
                        (blue<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (blue2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red>CONST.percentageOfTheSmallImageTakenByTheCrack*CONST.part1StrideX*CONST.part1StrideY)):#red
               
                    print("red:",red)
                    numpy.save(savePath+"numpy_arrays"+r"\red\0rot"+"\\"+imgName+"Red_"+str(countOfRedImages+1)+"_"+str(red)+".npy",arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX])
                    Image.fromarray(arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\red\0rot"+"\\"+imgName+"_Red_"+str(countOfRedImages+1)+"_"+str(red)+".jpg","JPEG")
                    Image.fromarray(arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\red\0rot"+"\\"+imgName+"_Red_"+str(countOfRedImages+1)+"_"+str(red)+"_noPaint.jpg","JPEG")
                    countOfRedImages+=1
            
                elif (
                    (green<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (green2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (yellow<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (yellow2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (red<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (red2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (blue2>CONST.percentageOfTheSmallImageTakenByTheCrack*CONST.part1StrideX*CONST.part1StrideY)):#blue
                
                    print("blue:",blue2)
                    numpy.save(savePath+"numpy_arrays"+r"\blue\0rot"+"\\"+imgName+"Blue2_"+str(countOfBlueImages+1)+"_"+str(blue2)+".npy",arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX])
                    Image.fromarray(arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\blue\0rot"+"\\"+imgName+"_Blue2_"+str(countOfBlueImages+1)+"_"+str(blue2)+".jpg","JPEG")
                    Image.fromarray(arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\blue\0rot"+"\\"+imgName+"_Blue2_"+str(countOfBlueImages+1)+"_"+str(blue2)+"_noPaint.jpg","JPEG")
                    countOfBlueImages+=1
                elif (
                        (green<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (green2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (yellow<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (yellow2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (blue>CONST.percentageOfTheSmallImageTakenByTheCrack*CONST.part1StrideX*CONST.part1StrideY)):#blue
                
                    print("blue:",blue)
                    numpy.save(savePath+"numpy_arrays"+r"\blue\0rot"+"\\"+imgName+"Blue_"+str(countOfBlueImages+1)+"_"+str(blue)+".npy",arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX])
                    Image.fromarray(arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\blue\0rot"+"\\"+imgName+"_Blue_"+str(countOfBlueImages+1)+"_"+str(blue)+".jpg","JPEG")
                    Image.fromarray(arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\blue\0rot"+"\\"+imgName+"_Blue_"+str(countOfBlueImages+1)+"_"+str(blue)+"_noPaint.jpg","JPEG")
                    countOfBlueImages+=1
            
                elif (
                        (yellow<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (yellow2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (red2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (blue<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (blue2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                        (green2>CONST.percentageOfTheSmallImageTakenByTheCrack*CONST.part1StrideX*CONST.part1StrideY)):#green
                
                    print("green:",green)
                    numpy.save(savePath+"numpy_arrays"+r"\green\0rot"+"\\"+imgName+"Green2"+str(countOfGreenImages+1)+"_"+str(green2)+".npy",arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX])
                    Image.fromarray(arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\green\0rot"+"\\"+imgName+"_Green2_"+str(countOfGreenImages+1)+"_"+str(green2)+".jpg","JPEG")
                    Image.fromarray(arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\green\0rot"+"\\"+imgName+"_Green2_"+str(countOfGreenImages+1)+"_"+str(green2)+"_noPaint.jpg","JPEG")
                    countOfGreenImages+=1
                elif (
                    (yellow<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (yellow2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (red<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (red2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (blue<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (blue2<CONST.percentageOfTheSmallImageTakenByNOnPrimaryCrack*CONST.part1StrideX*CONST.part1StrideY) and 
                    (green>CONST.percentageOfTheSmallImageTakenByTheCrack*CONST.part1StrideX*CONST.part1StrideY)):#green
                
                    print("green:",green)
                    numpy.save(savePath+"numpy_arrays"+r"\green\0rot"+"\\"+imgName+"Green"+str(countOfGreenImages+1)+"_"+str(green)+".npy",arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX])
                    Image.fromarray(arrayPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\green\0rot"+"\\"+imgName+"_Green_"+str(countOfGreenImages+1)+"_"+str(green)+".jpg","JPEG")
                    Image.fromarray(arrayNoPaint[y:y+CONST.part1StrideY,x:x+CONST.part1StrideX]).save(savePath+"cut_images"+r"\green\0rot"+"\\"+imgName+"_Green_"+str(countOfGreenImages+1)+"_"+str(green)+"_noPaint.jpg","JPEG")
                    countOfGreenImages+=1
 
                x+=CONST.part1StrideX
        
            x=0
            y=y+CONST.part1StrideY 


    if ("COLOR LINES" not in pathPaint) or ("COLOR LINES" in pathNoPaint):
        raise Exception("Check the paths! They lead to wrong images. paint:{} noPaint{}".format(pathPaint,pathNoPaint))


    
    #resizing the images
    imgFull = Image.open(pathPaint,mode="r").convert("RGB")
    imgResized = imgFull.resize((CONST.rescaleImageX,CONST.rescaleImageY))
    arrayPaint = numpy.asarray(imgResized)
    imgFullNoPaint = Image.open(pathNoPaint)
    imgFullNoPaintResized =imgFullNoPaint.resize((CONST.rescaleImageX,CONST.rescaleImageY)) 
    arrayNoPaint = numpy.asarray(imgFullNoPaintResized)
    
    cutTheimagesCreateArrays(arrayPaint,arrayNoPaint)

def creatingDatasetFromColoredImages(folderName):
    '''
    generates numpy arrays from the pained images in 4 easy steps
    1 - get the names of colored and not colored images
    2 - create the numpy arrays of the cuts of the images that have cracks
    3 - create the numpy arrays of the cuts of the images that don't have cracks
    4 - create augmented data images(rotated/reflected and different combinations of those) 
    '''

    def findAPaintedPairForNotPaintedImage(originalNames,paintedNames):
        '''
        used to return pairs of pained and not pained images
        '''
        def compare2strings(a:str,b:str):
            '''
            used in generating images to find the right names of the images
            '''
            count = 0
            for letNum in range(len(a)):
                if a[letNum] is b[letNum]:
                    count+=1
                else:
                    break
            return count

        out = {}
        for orStr in originalNames:
            dist =0
            pairedStr = paintedNames[0]
            for paintedStr in paintedNames:
                newDist = compare2strings(orStr,paintedStr)
                if newDist>dist:
                    dist = newDist 
                    pairedStr = paintedStr
        
            out[orStr] = pairedStr

        return out

    #1 step
    dir  = os.fsencode(CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\")

    notPaintedNames = []
    paintedNames = []

    notPaintedPaintedPairs = {}
    
    # get names of images without cracks and pair them with images with cracks
    print("Started generating images with cracks")
    for file in os.listdir(dir):
        fileName = os.fsdecode(file)
        if ".jpg" in fileName:
            if CONST.colored_and_notColored_img_separator in fileName:
                paintedNames.append(fileName)
            else:
                notPaintedNames.append(fileName)

    
    notPaintedPaintedPairs = findAPaintedPairForNotPaintedImage(notPaintedNames,paintedNames)


    #2 step
    #find and cut images with cracks
    if not CONST.coloredImagesAreCutAndNumpyArraysCreated:
        threads = [None] * len(notPaintedPaintedPairs)
        threadCount=0

        for k in notPaintedPaintedPairs:
            print(k)
            if CONST.useMiltithreading:
                threads[threadCount]  = threading.Thread(target=imageToNumpyArrays,args = (CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\"+notPaintedPaintedPairs[k],
                                                                                               CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\"+k,
                                                                                               CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\",
                                                                                               folderName+"_"+k[:-4],
                                                                                               ),daemon=True)
        
                threads[threadCount].start()
                threadCount+=1
            else:
                imageToNumpyArrays(CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\"+notPaintedPaintedPairs[k],
                                                                                               CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\"+k,
                                                                                               CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\",
                                                                                               folderName+"_"+k[:-4],
                                                                                               )
    
    if CONST.useMiltithreading:
        for t in threads:
            t.join()

    #3 step
    #generate images without cracks
    
    #get the number of images to generate//used only when the images already generated
    numOfCrackImagesGenerated = sum([len(files) for r, d, files in os.walk(CONST.pathToColoredImagesRootFolder+"//"+folderName+"//numpy_arrays//")])
    numImagesTogeneratePerImage =numOfCrackImagesGenerated//len(notPaintedNames)
    print(numOfCrackImagesGenerated)

    print("Srtarted sampleing images without cracks")
    if not CONST.theNoCrackDatasetIsCreated:
        for k in notPaintedPaintedPairs:
            print(k[:-4])
            if CONST.useMiltithreading:
                th = threading.Thread(target = CreateImagesWithoutCracks,args = (CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\"+notPaintedPaintedPairs[k],
                                                                             CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\"+k,numImagesTogeneratePerImage,
                                                                             CONST.pathToColoredImagesRootFolder+"\\"+folderName,folderName+k[:-4]),daemon = True)
                th.start()
            else:
                CreateImagesWithoutCracks(CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\"+notPaintedPaintedPairs[k],
                                                                             CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\"+k,numImagesTogeneratePerImage,
                                                                             CONST.pathToColoredImagesRootFolder+"\\"+folderName+"\\",folderName+k[:-4])
    
    #4 step 
    #create the augmented data
    #generate augmented images

    print("Started creating augmented type images")
    if not CONST.augmentedDataCreated:
        for color in CONST.listOfColorsUsed:
            pathDir =  CONST.pathToColoredImagesRootFolder+"\\"+folderName+ r"\numpy_arrays"+"\\"+color
            dir = os.fsencode(pathDir+r"\0rot"+"\\")
            for file in os.listdir(dir):
                fileName = os.fsdecode(file)
                saveAugmentedImages(fileName[:-4],pathDir+r"\0rot"+"\\"+fileName,pathDir)



#used for loading dataset
def getMyImagesList(listOfColors,folderNames,pathToFoldersDir,augTypesList):
    '''
    create lists of names and other info of the created dataset
    '''
    out={}
    for folder in folderNames:
        for color in listOfColors:
            for augType in augTypesList:
                directory = os.fsencode(pathToFoldersDir+"\\"+folder+r"\numpy_arrays"+"\\"+color+"\\"+augType+"\\")
                print(directory)
                if augType == "0rot":
                    for file in os.listdir(directory):
                        fn = os.fsdecode(file)
                        out[pathToFoldersDir+"\\"+folder+r"\numpy_arrays"+"\\"+color+"\\"+augType+"\\" +fn]=[pathToFoldersDir+"\\"+folder+r"\numpy_arrays"+"\\"+color+"\\"+augType+"\\"+fn]
                        for aT in CONST.augTypes:
                            if aT!="0rot":
                                out[pathToFoldersDir+"\\"+folder+r"\numpy_arrays"+"\\"+color+"\\"+augType+"\\" +fn].append(\
                                    pathToFoldersDir+"\\"+folder+r"\numpy_arrays"+"\\"+color+"\\"+aT+"\\" +fn[:-4]+"_"+aT+".npy")
                                
    noCracksNames = [out[k] for k in out.keys() if "noCracks" in k]
    yellowCracksNames =  [out[k] for k in out.keys() if "Yellow" in k]
    redCracksNames =  [out[k] for k in out.keys() if "Red" in k]
    blueCracksNames = [out[k] for k in out.keys() if "Blue" in k]
    greenCracksNames =  [out[k] for k in out.keys() if "Green" in k]                
    return yellowCracksNames,redCracksNames,blueCracksNames,greenCracksNames,noCracksNames

class saveDataNamesAndSplitTrTestFunctor:
    def __init__(self,yellow=[],red=[],blue=[],green=[],no=[]):
        self.yellow = yellow
        self.red = red
        self.blue = blue
        self.green=green
        self.no = no
        pass
    def saveDataNamesAndSplitTrTest(self,listOfColorsUsed,folderNames,augTypes,saveFolderPath,cv):
        def reShapeInputPredictByPretrained(x):
            x = numpy.vstack([x,numpy.zeros((2,30,3))])
            x = numpy.vstack([numpy.swapaxes(x,0,1),numpy.zeros((2,32,3))])
            x  =numpy.swapaxes(x,0,1)
            x = numpy.reshape(x,(1,32,32,3))
            return numpy.squeeze(CONST.pretrainedNetPart1.predict(x/255.0))

        '''
        input: theTrainingColorNumber -1 for multiclass classifiacation / 1 - yellow ,2 - red, 3 - blue, 4 - green for binary classification
        '''
         #check if all the data5415 will be loaded
        if len(CONST.augTypes)!=8:#check if have all augmented datatypes
            raise Exception("The list of augmented datatypes is not full")
        if len(CONST.folderNames)!=5:#check if have all folders    
            #raise Exception("The list of folders is not full")
            pass
        if len(CONST.listOfColorsUsed)!=4:#check if have all colors
            #raise Exception("The list of colors used is not full")
            pass


        #loading data
        if not CONST.useCrossValidation:
            #when cross-validation is not used
            self.yellow,self.red,self.blue,self.green,self.no = getMyImagesList(listOfColorsUsed,folderNames,CONST.pathToColoredImagesRootFolder,augTypes)
            shuffle(self.yellow)
            shuffle(self.red)
            shuffle(self.blue)
            shuffle(self.green)
            #decreasing the representation of the noCracks type
            self.no = [[i[0]] for i in self.no]#removing the augmented data
            self.no = sample(self.no,8*(len(self.yellow)+len(self.red)+len(self.blue)+len(self.green)))#making the same length as all other classes together
            shuffle(self.no)



            #split data into train and test
            #DO WE NEED TO FLATTEN??????
    
            trYellow = self.yellow[:int(len(self.yellow)*CONST.part1trTestSplit)]
            testYellow = self.yellow[int(len(self.yellow)*CONST.part1trTestSplit):]
            trRed = self.red[:int(len(self.red)*CONST.part1trTestSplit)]
            testRed = self.red[int(len(self.red)*CONST.part1trTestSplit):]
            trBlue = self.blue[:int(len(self.blue)*CONST.part1trTestSplit)]
            testBlue = self.blue[int(len(self.blue)*CONST.part1trTestSplit):]
            trGreen = self.green[:int(len(self.green)*CONST.part1trTestSplit)]
            testGreen = self.green[int(len(self.green)*CONST.part1trTestSplit):]                 
            trNo = self.no[:int(len(self.no)*CONST.part1trTestSplit)]
            testNo = self.no[int(len(self.no)*CONST.part1trTestSplit):]

            #print(testNo[0])
        else:
            #when we use cross-validation

            folder = r"S:\convnet_smaller_images\ConvNet\complete dataset\colored_images_and_np_arrays_from_them\dataNamesAndLabelsArrays\shuffled_separated_data_for_cv"

            if cv==0:
                self.yellow,self.red,self.blue,self.green,self.no = getMyImagesList(listOfColorsUsed,folderNames,CONST.pathToColoredImagesRootFolder,augTypes)
                shuffle(self.yellow)
                shuffle(self.red)
                shuffle(self.blue)
                shuffle(self.green)
                shuffle(self.no)
                

                
                with open(folder+r"\yellow.pickle", 'wb') as fp:
                    pickle.dump(self.yellow, fp)
                with open(folder+r"\red.pickle", 'wb') as fp:
                    pickle.dump(self.red, fp)
                with open(folder+r"\blue.pickle", 'wb') as fp:
                    pickle.dump(self.blue, fp)
                with open(folder+r"\green.pickle", 'wb') as fp:
                    pickle.dump(self.green, fp)
                with open(folder+r"\no.pickle", 'wb') as fp:
                    pickle.dump(self.no, fp)
            else:
                with open(folder+r"\yellow.pickle", 'rb') as fp:
                    self.yellow = pickle.load(fp)
                with open(folder+r"\red.pickle", 'rb') as fp:
                    self.red = pickle.load(fp)
                with open(folder+r"\blue.pickle", 'rb') as fp:
                    self.blue = pickle.load(fp)
                with open(folder+r"\green.pickle", 'rb') as fp:
                    self.green = pickle.load(fp)
                with open(folder+r"\no.pickle", 'rb') as fp:
                    self.no = pickle.load(fp)

            yellow = self.yellow
            red=self.red
            blue = self.blue
            green = self.green
            no = self.no
                
            trYellow = yellow[:int(len(yellow)*cv*CONST.part1trTestSplit)] + yellow[int(len(yellow)*(cv+1)*CONST.part1trTestSplit):]
            testYellow = yellow[int(len(yellow)*cv*CONST.part1trTestSplit):int(len(yellow)*(cv+1)*CONST.part1trTestSplit)]
            trRed = red[:int(len(red)*cv*CONST.part1trTestSplit)]+red[int(len(red)*(cv+1)*CONST.part1trTestSplit):]
            testRed = red[int(len(red)*cv*CONST.part1trTestSplit):int(len(red)*(cv+1)*CONST.part1trTestSplit)]
            trBlue = blue[:int(len(blue)*cv*CONST.part1trTestSplit)]+blue[int(len(blue)*(cv+1)*CONST.part1trTestSplit):]
            testBlue = blue[int(len(blue)*cv*CONST.part1trTestSplit):int(len(blue)*(cv+1)*CONST.part1trTestSplit)]
            trGreen = green[:int(len(green)*cv*CONST.part1trTestSplit)]+green[int(len(green)*(cv+1)*CONST.part1trTestSplit):]
            testGreen = green[int(len(green)*cv*CONST.part1trTestSplit):int(len(green)*(cv+1)*CONST.part1trTestSplit)]
            trNo = no[:int(len(no)*cv*CONST.part1trTestSplit)]+no[int(len(no)*(cv+1)*CONST.part1trTestSplit):]
            testNo = no[int(len(no)*cv*CONST.part1trTestSplit):int(len(no)*(cv+1)*CONST.part1trTestSplit)]
        

        trData = trYellow
        trData.extend(trRed)
        trData.extend(trBlue)
        trData.extend(trGreen)
        trData.extend(trNo)
        trData = [j for i in trData for j in i]
        shuffle(trData)

        testData = testYellow
        testData.extend(testRed)
        testData.extend(testBlue)
        testData.extend(testGreen)
        testData.extend(testNo)
        testData = [j for i in testData for j in i]
        shuffle(testData)

        #if we use output of the pretrained model
        if CONST.part1usePretrainedNetwork:
            trDataPretrained = []
            testDataPretrained = []
            import ntpath
            for i,v in enumerate(trData):
                ar = reShapeInputPredictByPretrained(numpy.load(v))
                fn = v.replace(CONST.pathToColoredImagesRootFolder,"").split("\\")[-1]
                numpy.save(CONST.part1pathToDataGeneratedByPretrainedNet+"\\"+fn,ar)
                trDataPretrained.append(CONST.part1pathToDataGeneratedByPretrainedNet+"\\"+fn)
            for i,v in enumerate(testData):
                ar = reShapeInputPredictByPretrained(numpy.load(v))
                fn = v.replace(CONST.pathToColoredImagesRootFolder,"").split("\\")[-1]
                numpy.save(CONST.part1pathToDataGeneratedByPretrainedNet+"\\"+fn,ar)
                testDataPretrained.append(CONST.part1pathToDataGeneratedByPretrainedNet+"\\"+fn)
            trData = trDataPretrained
            testData = testDataPretrained

        #creating labels
        trLabels = {}
        for img in trData: 
            if "noCracks" in img:
                trLabels[img] = CONST.numNoCrack
            if "Yellow" in img:
                trLabels[img] = CONST.numYellowCrack
            if "Red" in img:
                trLabels[img] = CONST.numRedCrack
            if "Blue" in img:
                trLabels[img] = CONST.numBlueCrack
            if "Green" in img:
                trLabels[img] = CONST.numGreenCrack

        testLabels = {}
        for img in testData: 
            if "noCracks" in img:
                testLabels[img] = CONST.numNoCrack
            if "Yellow" in img:
                testLabels[img] = CONST.numYellowCrack
            if "Red" in img:
                testLabels[img] = CONST.numRedCrack
            if "Blue" in img:
                testLabels[img] = CONST.numBlueCrack
            if "Green" in img:
                testLabels[img] = CONST.numGreenCrack
        
        #saving the data        
        numpy.save(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\trDataNames.npy",trData)
        with open(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\trLabels.json", 'w') as fp:
            json.dump(trLabels,fp)
        numpy.save(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\testDataNames.npy",testData)
        with open(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\testLabels.json", 'w') as fp:
            json.dump(testLabels,fp)
    


if __name__=="__main__":
    if not CONST.theDataSetFromColoredCracksIsGenerated:
        for folderName in CONST.folderNames:
            if CONST.useMiltithreading:
                th = threading.Thread(target = creatingDatasetFromColoredImages,args = (folderName),daemon = True)
                th.start()
            else:
                creatingDatasetFromColoredImages(folderName)
    
    saveDataNamesAndSplitTrTestFunctor().saveDataNamesAndSplitTrTest(CONST.listOfColorsUsed,CONST.folderNames,CONST.augTypes,CONST.part1ArraysOFDataNamesAndLabelsPath,0)

            