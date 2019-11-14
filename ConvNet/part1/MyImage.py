class MyImage:
    """larger image name , grid posidtion on the larger image, has the crack or no """
    dir = ""
    name = ""
    #grid_position_x = 0
    #grid_position_y = 0
    color = ""
    folder = ""
    def __init__(self,name,color,dir="",folder =""):
        self.name = name
        self.color = color
        self.dir = dir
        self.folder = folder


