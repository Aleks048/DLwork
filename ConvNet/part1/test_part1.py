import preprocessingPaintedCracks
import CONSTANTS as CONST
import numpy

#preprocessingPaintedCracks.imageToNumpyArrays(r"S:\convnet_smaller_images\ConvNet\complete dataset\colored_images_and_np_arrays_from_them\4\ABD D15 R0 A COLOR LINES.jpg",
#                                              r"S:\convnet_smaller_images\ConvNet\complete dataset\colored_images_and_np_arrays_from_them\4\ABD D15 R0 A.jpg",
#                                              "",
#                                              ""
#                                              )


#development of using VGG19 pretrained
#ar = numpy.load(r"S:\convnet_smaller_images\ConvNet\complete dataset\colored_images_and_np_arrays_from_them\1\numpy_arrays\noCracks\0rot\1ABC A01 R0A (camera 305)noCracks_1.npy")/255.0
#arAdd = numpy.vstack([ar,numpy.zeros((2,30,3))])
#arAdd = numpy.vstack([numpy.swapaxes(arAdd,0,1),numpy.zeros((2,32,3))])
#arAdd  =numpy.swapaxes(arAdd,0,1)
#arAdd = numpy.reshape(arAdd,(1,32,32,3))
##arAdd = numpy.vstack([numpy.transpose(arAdd),numpy.zeros((2,32,3))])
#print(numpy.shape(arAdd))

#print(CONST.pretrainedNetPart1.predict(arAdd))


