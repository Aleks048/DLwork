from PIL import Image
import numpy
import keras
import time

print("Bom Dia Marcelo. My name is HAL 9000.")
print("I can classify images of the size 180x180 pixels in two classes:")
print("    1) the image has a crack of type YY1")
print("    2) the image does NOT have the crack YY1")

while True:
    modelPath = input("Please input the full path of your model file:")
    try:
        Image.open(modelPath)
        break
    except IOError:
        print("The path is incorrect. Sorry.")
classificationOut = []
outFile=input("Please input the name of the output file: ")
print("The file "+outFile+".txt with the experiment results will be saved in the output directory.")
while True:
    while True:
        imagePath = input("Please input the full path of your test image:")
        try:
            im = Image.open(imagePath)
            break
        except IOError:
            print("The path is incorrect. Sorry.")


    imageArr = numpy.expand_dims(numpy.asarray(im),axis=0)


    model = keras.models.load_model(modelPath)
    prediction = model.predict(imageArr,verbose=0)
    if prediction[0,0]<prediction[0,1]:
        print("The image is of class 1.")
        print(imagePath+" has crack of type YY1")
        classificationOut.append(imagePath+" has crack of type YY1")
    else:
        print("The image is of class 2.")
        print(imagePath+" has NO crack of type YY1")
        classificationOut.append(imagePath+" has no crack of type YY1")
    exitCheck = input("PLease input E to exit ot anything else to continue: ")
    if exitCheck=="E":
        break
with open("../outputFiles/"+outFile+'.txt', 'w') as f:
    for item in classificationOut:
        f.write("%s\n" % item)
