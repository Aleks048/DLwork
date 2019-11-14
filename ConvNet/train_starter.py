import os

colors = ["yellow","red","blue","green"]

numOfLearningRates = 999
#for j in range(0,numOfLearningRates,40):
   
#    #os.system(r'python S:\convnet_smaller_images\ConvNet\ConvNet\theMainTrainingScript.py '+str(colors[j])+" "+str(j))
#    os.system(r'python S:\convnet_smaller_images\ConvNet\ConvNet\large_dataset_preprocessing_for_dri_prediction.py '+"1."+str(j)+"e-4")
#    pass
for j in range(0,numOfLearningRates,40):
   
    #os.system(r'python S:\convnet_smaller_images\ConvNet\ConvNet\theMainTrainingScript.py '+str(colors[j])+" "+str(j))
    os.system(r'python S:\convnet_smaller_images\ConvNet\ConvNet\large_dataset_preprocessing_for_dri_prediction.py '+"1."+("0"+str(j) if j<100 else str(j)) +"e-5")
    pass
for j in range(0,numOfLearningRates,40):
   
    #os.system(r'python S:\convnet_smaller_images\ConvNet\ConvNet\theMainTrainingScript.py '+str(colors[j])+" "+str(j))
    os.system(r'python S:\convnet_smaller_images\ConvNet\ConvNet\large_dataset_preprocessing_for_dri_prediction.py '+"1."+str(j)+"e-3")
    pass