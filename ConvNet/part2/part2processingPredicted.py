#when removing the second coordinate is needed
        #trRemovedSecondValue = numpy.empty((4,len(trData[0]),60,60,1))
        #for l in range(4):
        #    for i in range(len(trData[0])):
        #        for j in range(60):
        #            for k in range(60):
        #                trRemovedSecondValue[l][i][j][k] = trData[l][i][j][k][0]
        #numpy.save(r"S:\convnet_smaller_images\convnet\painted_data\large_dataset\trainingDataSecondValueRemoved.npy",trRemovedSecondValue)

        #thresholding
        #trRemovedSecondValue = numpy.empty((4,len(trData[0]),60,60,1))
        #for l in range(4):
        #    for i in range(len(trData[0])):
        #        for j in range(60):
        #            for k in range(60):
        #                trRemovedSecondValue[l][i][j][k] = 1 if trData[l][i][j][k][0]>0.9 else 0
        #numpy.save(r"S:\convnet_smaller_images\convnet\painted_data\large_dataset\trainingThreshold0_9DataSecondValueRemoved.npy",trRemovedSecondValue)
        
        #trRemovedSecondValue = numpy.empty((4,len(trData[0]),60,60,1))
        #for l in range(4):
        #    for i in range(len(trData[0])):
        #        for j in range(60):
        #            for k in range(60):
        #                trRemovedSecondValue[l][i][j][k] = 1 if trData[l][i][j][k][0]>0.99 else 0
        #numpy.save(r"S:\convnet_smaller_images\convnet\painted_data\large_dataset\trainingThreshold0_99DataSecondValueRemoved.npy",trRemovedSecondValue)
        
        #trRemovedSecondValue = numpy.empty((4,len(trData[0]),60,60,1))
        #for l in range(4):
        #    for i in range(len(trData[0])):
        #        for j in range(60):
        #            for k in range(60):
        #                trRemovedSecondValue[l][i][j][k] = 1 if trData[l][i][j][k][0]>0.5 else 0
        #numpy.save(r"S:\convnet_smaller_images\convnet\painted_data\large_dataset\trainingThreshold0_5DataSecondValueRemoved.npy",trRemovedSecondValue)

        #trRemovedSecondValue = numpy.empty((4,len(trData[0]),60,60,1))
        #for l in range(4):
        #    for i in range(len(trData[0])):
        #        for j in range(60):
        #            for k in range(60):
        #                trRemovedSecondValue[l][i][j][k] = 1 if trData[l][i][j][k][0]>0.7 else 0
        #numpy.save(r"S:\convnet_smaller_images\convnet\painted_data\large_dataset\trainingThreshold0_7DataSecondValueRemoved.npy",trRemovedSecondValue)

        #trRemovedSecondValue = numpy.empty((4,len(trData[0]),60,60,1))
        #for l in range(4):
        #    for i in range(len(trData[0])):
        #        for j in range(60):
        #            for k in range(60):
        #                trRemovedSecondValue[l][i][j][k] = 1 if trData[l][i][j][k][0]>0.3 else 0
        #numpy.save(r"S:\convnet_smaller_images\convnet\painted_data\large_dataset\trainingThreshold0_3DataSecondValueRemoved.npy",trRemovedSecondValue)

        #summing up
        #trRemovedSecondValue = numpy.empty((4,len(trData[0])))
        #for c in range(4):
        #    for i in range(len(trLabels)):
        #        sum = 0
        #        for j in range(60):
        #            for k in  range(60):
        #                sum+=trData[c][i][j][k]
        #        trRemovedSecondValue[c][i] = sum
        #numpy.save(r"S:\convnet_smaller_images\convnet\painted_data\large_dataset\trainingSummed_DataSecondValueRemoved.npy",trRemovedSecondValue)
