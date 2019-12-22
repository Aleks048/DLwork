import keras.utils
import numpy
import CONSTANTS as CONST

class DataGenerator(keras.utils.Sequence):
    '''
    the class is used to feed data into the machines of part 1
    '''
    def __init__(self,list_IDs,labels,batch_size=95,dim = (227,227),n_channels=3,n_classes=2,shuffle=True,useConv=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.useConv = useConv
        self.on_epoch_end()
    
    def on_epoch_end(self):
        #print(len(self.list_IDs))
        self.indexes = numpy.arange(len(self.list_IDs))
        if self.shuffle == True:
            numpy.random.shuffle(self.indexes)  
        
    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      
      if self.useConv:
        X = numpy.empty((self.batch_size, *self.dim, self.n_channels))
      else:
        X = numpy.empty((self.batch_size,self.dim[0]))
      y = numpy.empty((self.batch_size), dtype=int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          X[i,] = numpy.load(ID)

          #print(ID)
          # Store class
          y[i] = self.labels[ID]
    
      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
      
    def __len__(self):
         'Denotes the number of batches per epoch'
         return int(numpy.floor(len(self.list_IDs) // self.batch_size))

    def __getitem__(self, index):
          'Generate one batch of data'
          # Generate indexes of the batch
          indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
          
          # Find list of IDs
          list_IDs_temp = [self.list_IDs[k] for k in indexes]

          
          # Generate data
          X, y = self.__data_generation(list_IDs_temp)

          return X, y

    

