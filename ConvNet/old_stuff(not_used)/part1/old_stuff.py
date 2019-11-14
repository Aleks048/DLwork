#genrate the original images numpy arrays of images with cracks
#folder1
#for folderName in folderNames:
#    for imageNum in range(numOfImagesPerFolder[folderName]):
#        print("imageNum: "+str(imageNum+1))
#        print(datetime.datetime.time(datetime.datetime.now()))
#        imageToNumpyArrays(r"C:\Users\ytr16\source\repos\ConvNet\painted_data"+"\\"+folderName+"\\"+"ABC A"+("0"+str(imageNum+1)if(imageNum+1<10)else(str(imageNum+1)))+r" R0A (camera 305) - COLOR LINES.jpg",r"C:\Users\ytr16\source\repos\ConvNet\painted_data"+"\\"+folderName+"\\"+"ABC A"+("0"+str(imageNum+1)if(imageNum+1<10)else(str(imageNum+1)))+r" R0A (camera 305).jpg",r"C:\Users\ytr16\source\repos\ConvNet\painted_data"+"\\"+str(folderName)+r"\numpy_arrays","imgFol"+str(folderName)+"ImgNum"+str(imageNum+1))
#folder2
#for folderName in folderNames:
#    path = r"C:\Users\ytr16\source\repos\ConvNet\painted_data\\"
#    dir = os.fsencode(path+folderName+"\\")
#    for file in os.listdir(dir):
#        fileName = os.fsdecode(file)
#        #imageToNumpyArrays(path+folderName+"\\"+"ABV A04 R0A - 4 J + C 35 - 16 - 0.30_.jpg",path+folderName+"\\"+"ABV A04 R0A.jpg",path+folderName+"\\"+"numpy_arrays",folderName)
#        if "+" in fileName:
#            print("imageName: "+fileName)
#            print(datetime.datetime.time(datetime.datetime.now()))
#            imageToNumpyArrays(path+folderName+"\\"+fileName,path+folderName+"\\"+fileName[:-30]+".jpg",path+folderName+"\\"+"numpy_arrays",folderName+fileName[:-4])

#folder3 & 4 potentially universal
#for folderName in  folderNames:
#    path = r"S:\ConvNet\painted_data\\"
#    dir = os.fsencode(path+folderName+"\\")
    
#    originalNames = []
#    paintedNames = []
#    pairNames = {}
#    print(dir) 
#    for file in os.listdir(dir):
#        filename = os.fsdecode(file)
#        if ".jpg" in filename:
#            if numOfImagesPerFolder[folderName].keyword in filename:
#                paintedNames.append(filename)
#            else:
#                originalNames.append(filename)
    
    
#    for orStr in originalNames:
#        dist = compare2strings(orStr,paintedNames[0])
#        pairedStr = paintedNames[0]
#        for paintedStr in paintedNames:
#           newDist =compare2strings(orStr,paintedStr)
#           if newDist>dist:
#               dist = newDist 
#               pairedStr = paintedStr

#        pairNames[orStr] = pairedStr


#    for k in pairNames:
#        imageToNumpyArrays(path+folderName+"\\"+pairNames[k],path+folderName+"\\"+k,path+folderName+"\\"+"numpy_arrays",folderName+"_"+k[:-4])
        

#generate augmented data

#for folderName in folderNames:
#    for color in listOfColorsUsed:
#        pathDir =  r"S:\ConvNet\painted_data"+"\\"+folderName+ r"\numpy_arrays"+"\\"+color
#        dir = os.fsencode(pathDir+r"\0rot"+"\\")
#        for file in os.listdir(dir):
#            #print(len(os.listdir(dir)))
#            fileName = os.fsdecode(file)
#            saveAugmentedImages(fileName[:-4],pathDir+r"\0rot"+"\\"+fileName,pathDir)


#sample images without cracks

#imagesWithCracksCount =0 
#for folderName in folderNames:
#    for color in listOfColorsUsed:
#        pathDir = r"S:\ConvNet\painted_data"+"\\"+folderName+ r"\numpy_arrays"+"\\"+color
#        dir = os.fsencode(pathDir+r"\0rot"+"\\")
#        imagesWithCracksCount+=len(os.listdir(dir))
#print(imagesWithCracksCount)

#folders 1 & 2
#for folderName in folderNames:
#    pathDir =  r"S:\ConvNet\painted_data\\"+"\\"+folderName+"\\"
#    dir = os.fsencode(pathDir)
#    for file in os.listdir(dir):
#        fileName = os.fsdecode(file)
#        if fileName[-4:]==".jpg":
#            if folderName=="1":
#                if "COLOR LINES" in os.fsdecode(file):#folder 1
#                    CreateImagesWithoutCracks(pathDir+fileName,pathDir+fileName[:-18]+".jpg",imagesWithCracksCount//(numOfImagesPerFolder[folderName].numImages),pathDir+"numpy_arrays\\",fileName[:-18])#fold
#            elif foldeName=="2":    
#                if "+" in os.fsdecode(file):#folder 2
#                    CreateImagesWithoutCracks(pathDir+fileName,pathDir+fileName[:-30]+".jpg",imagesWithCracksCount//(numOfImagesPerFolder[folderName].numImages),pathDir+"numpy_arrays\\",fileName[:-18])#folder2
#folders 3&4
#for folderName in folderNames:
#    path = r"S:\ConvNet\painted_data\\"
#    dir = os.fsencode(path+folderName+"\\")
#    originalNames = []
#    paintedNames = []
#    pairNames = {}
#    for file in os.listdir(dir):
#        filename = os.fsdecode(file)
#        if ".jpg" in filename:
#            if numOfImagesPerFolder[folderName].keyword in filename:
#                paintedNames.append(filename)
#            else:
#                originalNames.append(filename)    
#    for orStr in originalNames:
#        dist = compare2strings(orStr,paintedNames[0])
#        pairedStr = paintedNames[0]
#        for paintedStr in paintedNames:
#            newDist =compare2strings(orStr,paintedStr)
#            if newDist>dist:
#                dist = newDist 
#                pairedStr = paintedStr
#        pairNames[orStr] = pairedStr
#    for k in pairNames:
#        CreateImagesWithoutCracks(path+folderName+"\\"+pairNames[k],path+folderName+"\\"+k,imagesWithCracksCount//(numOfImagesPerFolder[folderName].numImages),path+folderName+"\\"+"numpy_arrays\\",folderName+"_"+k[-4])
##augtypes for noCracks
#for folderName in folderNames:
#    dirPathNoCracks =  r"S:\ConvNet\painted_data"+"\\"+folderName+r"\numpy_arrays\noCracks"
#    directory = os.fsencode(dirPathNoCracks+"\\0rot\\")
#    for file in os.listdir(directory):
#        fileName = os.fsdecode(file)
#        saveAugmentedImages(fileName[:-4],dirPathNoCracks+"\\0rot\\"+fileName,dirPathNoCracks)


##getMyIMagesList(listOfColorsUsed,folderNames,r"C:\Users\ytr16\source\repos\ConvNet\painted_data\\",augTypes)
