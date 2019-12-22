'''
store all the funcs related to the log here
'''
import part1.CONSTANTS_part1 as CONST
def findHowManyImagesInEachClass(data,dataType:str):
    '''
    given the data prints the number of images in each of 5 classes:
        - Yellow
        - Red
        - Blue
        - Green
        - noCracks
    '''
    counts = [0,0,0,0]
    for x in data:
        if "Yellow" in x:
            counts[CONST.numYellowCrack]+=1
        if "Red" in x:
            counts[CONST.numRedCrack]+=1
        if "Blue" in x:
            counts[CONST.numBlueCrack]+=1
        if "Green" in x:
            counts[CONST.numGreenCrack]+=1
        if "noCrack" in x:
            counts[CONST.numNoCrack]+=1
    print(counts)
    print("The "+dataType+" data distribution:",[i/sum(counts) for i in counts])
    print()
