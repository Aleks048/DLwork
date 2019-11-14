from PIL import Image

import xlrd

import os
import gc
import numpy

from MyImage import MyImage

#chacks if the image is in the list
def checkImagesInList(imagesList,name):
    for img in imagesList:
        if (img.name == name):
            return True
    return False

#creates .npy array from image
def gridArraysFromImage(addressOriginal,nameOriginal,adressSave,numOfGrids,width,height,imagesWithCracksList):
     mainImg = Image.open(addressOriginal+"\\"+nameOriginal+".jpg")

     mainImgArr=numpy.empty((0,width,height,3))

     gridWidth = width//numOfGrids
     gridHeight = height//numOfGrids

     count = 0

     mainImgArr = numpy.asarray(mainImg)
     for y in range(1,numOfGrids+1):
        for x in range(1,numOfGrids+1):
           
            nameAdd ="_X-"+(str(x) if (x>=10) else ("0"+str(x))) +"_Y-"+(str(y) if ((y)>=10) else ("0"+str(y)))
            
            if checkImagesInList(imagesWithCracksList,nameOriginal+nameAdd): 
                gridArr = mainImgArr[(x-1)*gridWidth:(x)*gridWidth,(y-1)*gridHeight:(y)*gridHeight]
                gridImg = Image.fromarray(gridArr)
                str_x = str(x)if(x>=10)else ("0"+str(x))
                str_y = str(y)if(y>=10)else ("0"+str(y))
                #print(adressSave+nameOriginal+"_X-"+str_x+"_Y-"+str_y+".npy")
                numpy.save(adressSave+nameOriginal+"_X-"+str_x+"_Y-"+str_y+".npy",gridArr)
                count+=1
         
     #print(count)
     #print()
     mainImg.close()

#creates .npy arrays from list of images
def createGridArraysFromImages(readyFolderNames,imgWidth,imgHeight,numOfGrids,imagesWithCracksList):
    for folderName in readyFolderNames:
        originalFileDirStr =r"C:\Users\ytr16\source\repos\ConvNet\grid_data\180"+"\\"+folderName+r"\original"
        originalFileDir=os.fsencode(originalFileDirStr)
        for file in os.listdir(originalFileDir):
            if not(("grid" in str(file[:])) or ("Grid" in str(file[:])) or ("GRID" in str(file[:]))):
                fileNameBase = file[:-4].decode('ascii')
                
                gridArraysFromImage(originalFileDirStr,fileNameBase,
                         r"C:\Users\ytr16\source\repos\ConvNet\grid_data\180"+"\\"+folderName+r"\train\0rot\numpyArrays"+"\\",
                         numOfGrids,imgWidth,imgHeight,imagesWithCracksList)

#list of images with cracks
def getTheImagesWithCracksList(readyFolders,numOfImagesPerDir,augType):
    excelFile = xlrd.open_workbook('..\\grid_data\\180\\Codes.xlsx')
    outImages = []


    for folderName in readyFolders:
        worksheet = excelFile.sheet_by_name("ACE "+folderName+"01 TO "+folderName+"15")
        row = 4
        coloumn = 4
        for numOfIm in range(numOfImagesPerDir):
            nameBase = worksheet.cell(row,0).value[:-1]+"_"+worksheet.cell(row,1).value+"_"+worksheet.cell(row,2).value
            
            dir = r"C:\Users\ytr16\source\repos\ConvNet\grid_data\180"+"\\"+folderName+"\\train\\"+augType+"\\numpyArrays\\"
            while True:
                while (worksheet.cell(row,coloumn).value!=xlrd.empty_cell.value):
                
                    grid_x = int(worksheet.cell(row,coloumn).value)
                    grid_y = int(worksheet.cell(row,coloumn+1).value)

                    nameAdd ="_X-"+(str(grid_x) if ((grid_x)>=10) else ("0"+str(grid_x))) +"_Y-"+(str(grid_y) if ((grid_y)>=10) else ("0"+str(grid_y)))

                    img = MyImage(nameBase+nameAdd,worksheet.cell(row,coloumn).value,worksheet.cell(row,coloumn+1).value,worksheet.cell(row,3).value,dir)
                    if not(checkImagesInList(outImages,img.name)):
                        outImages.append(img)
                    coloumn+=3
                    
                   
                coloumn = 4
                
                if (worksheet.cell(row+1,coloumn).value==xlrd.empty_cell.value):
                    row+=2
                    break
                else:
                    row+=1
    return outImages     

#list of images without cracks
def createImagesWithoutCracksList(imagesWithCracksList,createGet,folderNames=[],augType=""):
    imagesWithoutCracksList=[]
    
   
    if createGet:
        for img in imagesWithCracksList:
            rand_grid_x = numpy.random.randint(1,11)
            rand_grid_y = numpy.random.randint(1,11)
            name = img.name[:-7]+(str(rand_grid_x) if (rand_grid_x>=10) else ("0"+str(rand_grid_x)))+img.name[-5:-2]+(str(rand_grid_y) if (rand_grid_y>=10) else ("0"+str(rand_grid_y)))

            if checkImagesInList(imagesWithCracksList,name) or checkImagesInList(imagesWithoutCracksList,name):
                while True:
                
                    rand_grid_x = numpy.random.randint(1,11)
                    rand_grid_y = numpy.random.randint(1,11)
                    name = img.name[:-7]+(str(rand_grid_x) if (rand_grid_x>=10) else ("0"+str(rand_grid_x)))+img.name[-5:-2]+(str(rand_grid_y) if (rand_grid_y>=10) else ("0"+str(rand_grid_y)))

                    if (not(checkImagesInList(imagesWithCracksList,name)) and not(checkImagesInList(imagesWithoutCracksList,name))):
                        break
        
            imgAppend = MyImage(name,float(rand_grid_x),float(rand_grid_y),"No",img.dir)
            imagesWithoutCracksList.append(imgAppend)
    else:
        for folderName in folderNames:
            fileDirStr =r"C:\Users\ytr16\source\repos\ConvNet\grid_data\180"+"\\"+folderName+r"\train"+"\\"+augType+"\\numpyArrays\\"
            fileDir=os.fsencode(fileDirStr)
            for file in os.listdir(fileDir):
                grid_x = float(file[14:-9].decode("ascii"))
                   
                grid_y = float(file[19:-4].decode("ascii"))
                name = file[:11].decode("ascii")+"_X-"+file[14:-9].decode("ascii")+"_Y-"+file[19:-4].decode("ascii")
                if not(checkImagesInList(imagesWithCracksList,name)):
                    imagesWithoutCracksList.append(MyImage(name,grid_x,grid_y,"No",fileDirStr))

    return imagesWithoutCracksList


def createAugmentedImages(readyFolders,augTypes):
    for folder in readyFolders:
        directoryTrain=os.fsencode(r"..\grid_data\180"+"\\"+folder+"\\train\\0rot\\numpyArrays\\")
        for file in os.listdir(directoryTrain):
            rotated0 = numpy.load((directoryTrain+file).decode("ascii"))
            #rotations arrays
                
            rotated90 = numpy.rot90(rotated0,axes=(-3,-2))
            rotated180 = numpy.rot90(rotated90,axes=(-3,-2))     
            rotated270 = numpy.rot90(rotated180,axes=(-3,-2))
            
            #reflections arrays
            reflection0 = numpy.fliplr(rotated0)
            reflection90 = numpy.fliplr(rotated90)
            reflection180 = numpy.fliplr(rotated180)
            reflection270 = numpy.fliplr(rotated270)
                
           
            
            #saving
            numpy.save(r"..\grid_data\180"+"\\"+folder+"\\train\\90rot\\numpyArrays\\"+file[:-4].decode("ascii")+".npy",rotated90)
            numpy.save(r"..\grid_data\180"+"\\"+folder+"\\train\\180rot\\numpyArrays\\"+file[:-4].decode("ascii")+".npy",rotated180)
            numpy.save(r"..\grid_data\180"+"\\"+folder+"\\train\\270rot\\numpyArrays\\"+file[:-4].decode("ascii")+".npy",rotated270)

            numpy.save(r"..\grid_data\180"+"\\"+folder+"\\train\\0ref\\numpyArrays\\"+file[:-4].decode("ascii")+".npy",reflection0)
            numpy.save(r"..\grid_data\180"+"\\"+folder+"\\train\\90ref\\numpyArrays\\"+file[:-4].decode("ascii")+".npy",reflection90)
            numpy.save(r"..\grid_data\180"+"\\"+folder+"\\train\\180ref\\numpyArrays\\"+file[:-4].decode("ascii")+".npy",reflection180)
            numpy.save(r"..\grid_data\180"+"\\"+folder+"\\train\\270ref\\numpyArrays\\"+file[:-4].decode("ascii")+".npy",reflection270)
#param
readyFolders = ["A","B","C","D"]
dataAugTypes = ["0rot"]

numOfImagesPerDir = 15

imgHeight = 1800
imgWidth =1800
numOfgrids = 10

#main activity

#CrackImgList = getTheImagesWithCracksList(readyFolders,numOfImagesPerDir,"0rot")


#NoCrackImgList = createImagesWithoutCracksList(CrackImgList,True)


##print(CrackImgList[11].name)
#print(len(NoCrackImgList))

#createGridArraysFromImages(readyFolders,imgWidth,imgHeight,numOfgrids,CrackImgList)

#createGridArraysFromImages(readyFolders,imgWidth,imgHeight,numOfgrids,NoCrackImgList)


#createAugmentedImages(readyFolders,dataAugTypes)
