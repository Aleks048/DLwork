'''
this class is used to store info about the painted images folders// like the keyword to separate painted images from non-painted ones,number of images in the folder
'''
class myFolder:
    numImages=0
    keyword = ""
    def __init__(self,numImages,keyword):
        self.keyword = keyword
        self.numImages = numImages