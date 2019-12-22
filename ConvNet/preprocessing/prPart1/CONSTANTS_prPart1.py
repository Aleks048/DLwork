'''
constants used for the preprocessing for 1
'''

'''creatingTheDataset'''
pathToColoredImagesRootFolder = "..\complete dataset\colored_images_and_np_arrays_from_them"#root of the dataset

useMiltithreading = False
coloredImagesAreCutAndNumpyArraysCreated = True
theNoCrackDatasetIsCreated = True
augmentedDataCreated = True
theDataSetFromColoredCracksIsGenerated = True
imagesCutArraysCreated = True

rejection_colour_parameters = {"Yellow":[range(245,256),range(245,256),range(97,107)],#used when finding the crack mask//if the pixel is in the range then it is a mask of certain color
                               "Yellow2":[range(200,256),range(200,256),range(0,45)],
                               "Red":[range(200,256),range(0,70),range(0,70)],
                               "Red2":[range(200,256),range(0,40),range(0,40)],
                               "Blue":[range(97,107),range(97,107),range(245,256)],
                               "Blue2":[range(0,40),range(0,40),range(200,256)],
                               "Green":[range(0,120),range(180,256),range(0,120)],
                               "Green2":[range(0,40),range(200,256),range(0,40)]}


coloredImageBoundary = 7#used to avoid cracks that appear close to the boundary of the cutted image

#theses are the parameters of that define how much of a crack must be in the image and how much of the crack of other type is allowed in the image
percentageOfTheSmallImageTakenByTheCrack = 0.04/3.5
percentageOfTheSmallImageTakenByNOnPrimaryCrack = 0.001

colored_and_notColored_img_separator = "COLOR LINES"#marker in the name of the data to separate image with mask and without it

