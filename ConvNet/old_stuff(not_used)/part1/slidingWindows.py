
import keras
import numpy
import xlrd
from PIL import Image

folderNames = ["AAA"]
numOfImages = 105



#for k in range(1):
def getImagesNames(numOfImages):

    imageNames={}
    

    excelFile = xlrd.open_workbook("C:\\Users\\ytr16\\source\\repos\\ConvNet\\grid_data\\180\\testing_images\\Codes.xlsx")

    for folderName in folderNames:
        worksheet = excelFile.sheet_by_name(folderName)
        for num in range(numOfImages):
            if worksheet.cell(num+4,4).value!=xlrd.empty_cell.value:
                if worksheet.cell(num+4,0).value[:3] =="AAA":
                    imName = worksheet.cell(num+4,0).value[:3]+"\images\\"+str(int(worksheet.cell(num+4,0).value[5:7]))+worksheet.cell(num+4,0).value[4:5]
                else:
                    imName = worksheet.cell(num+4,0).value[:3]+"\images\\"+worksheet.cell(num+4,0).value
                imageNames[imName]=1
            else:
                if worksheet.cell(num+4,0).value[:3] =="AAA":
                    imName = worksheet.cell(num+4,0).value[:3]+"\images\\"+str(int(worksheet.cell(num+4,0).value[5:7]))+worksheet.cell(num+4,0).value[4:5]
                else:
                    imName = worksheet.cell(num+4,0).value[:3]+"\images\\"+worksheet.cell(num+4,0).value
                imageNames[imName]=0
    
    return imageNames


def evaluateImage(imagePathAndName,model):
   

    fullImage = Image.open(imagePathAndName)
    #fullImageArr = numpy.empty((0,1800,1800,3))
    fullImageArr = numpy.asarray(fullImage)
    
    #print(fullImageArr.shape)

    y=0
    numOfImagesClassifiedAsCracks = 0
    imageCount=0

    predictions = []

    strideX = 1
    strideY = 45

    while y+180<=fullImageArr.shape[1]:
        x=0
        while x+180<=fullImageArr.shape[0]:
            imageSmall = numpy.expand_dims(fullImageArr[x:x+180,y:y+180],axis=0)
            
            
            #print(imageSmall.shape)
            #t = Image.fromarray(imageSmall)
            #t.show()
            
            prediction = model.predict(imageSmall,verbose=0)
            
            if (prediction[0,0]<prediction[0,1]):

                predictions.append(1)
                numOfImagesClassifiedAsCracks+=1
            else:
                predictions.append(0)
            #print(prediction)
            imageCount+=1
            x+=strideX
            
        y+=strideY
    
    fullImage.close()

    out = 0
    #out = 0 if (numOfImagesClassifiedAsCracks/imageCount)<0.015 else 1

    lengthOfClassifiedAsCrack = 300
    print(imagePathAndName)
    print(predictions)
    numberOfcracksFound =40

    for i in range(len(predictions)-lengthOfClassifiedAsCrack):
        for j in range(lengthOfClassifiedAsCrack):
            if predictions[i+j]==0:
                break
            elif (j==lengthOfClassifiedAsCrack-1):
                out += 1
        if out>numberOfcracksFound:
            break

    print(out)
    return 0 if (out==0) else (0 if out<numberOfcracksFound else 1)
    

def evaluateModel(folderNames,imageNames):
    model  = keras.models.load_model("C:\\Users\\ytr16\\source\\repos\\ConvNet\\grid_data\\180\\models\\model1.h5")
    #for folderName in folderNames:
    correctCount = 0
    count = 0
    for imageName in imageNames:
        #print("C:\\Users\\ytr16\\source\\repos\\ConvNet\\grid_data\\180\\testing_images\\"+imageName+".jpg")
        #print(imageName)
        #print(imageNames[imageName])
        predicted = evaluateImage("C:\\Users\\ytr16\\source\\repos\\ConvNet\\grid_data\\180\\testing_images\\"+imageName+".jpg",model)
        print("hi "+str(predicted))
        if predicted == imageNames[imageName]:
            correctCount+=1

        count+=1
        
    print(correctCount/count)
        
if __name__=="__main__":
    evaluateModel(folderNames,getImagesNames(numOfImages))

#evaluateImage("C:\\Users\\ytr16\\source\\repos\\ConvNet\\grid_data\\180\\testing_images\\AAA\\images\\6G.jpg")