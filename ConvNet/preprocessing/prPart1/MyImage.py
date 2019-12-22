'''
this class is used to store the info for each image
used when the images are split during preprocessing
'''
class MyImage:
    dir = ""#?
    name = ""#?
    color = ""#?
    folder = ""#?
    def __init__(self,name,color,dir="",folder =""):
        self.name = name
        self.color = color
        self.dir = dir
        self.folder = folder


