

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2


def open_grby( path, id): 
    '''a function that reads GRBY image'''
    suffs = ['green', 'red', 'blue','yellow']
    cvflag = cv2.IMREAD_GRAYSCALE    
    img = [cv2.imread(os.path.join( path, '{}_{}.png'.format(id, suff) ), cvflag).astype(np.float32)/255 
           for suff in suffs ]
    return np.stack(img, axis=-1)

def make_dataset( path, metadata, train=True):
    '''load file patch for disk
    '''
    data = pd.read_csv( os.path.join( path, metadata) )
    if train:
        def fill_targets(row):
            target = np.array(row.Target.split(" ")).astype(np.int)
            p = np.zeros( 28 )
            p[target] = 1  #/len(target)  #<- !!!! P(W,X)
            row.Target = p
            return row
        data = data.apply(fill_targets, axis=1)
    return data


class ATLASProvide( object ):
    '''Provide for ATLAS dataset
    '''
    @classmethod
    def create(
        cls, 
        path,
        train=True,
        folders_images='train',
        metadata='train.csv',
        ):
        '''
        Factory function that create an instance of ATLASProvide and load the data form disk.
        '''
        provide = cls(path, train, folders_images, metadata )
        return provide
    
    def __init__(self,
        path,        
        train=True,
        folders_images='train',
        metadata='train.csv',
        ):
        super(ATLASProvide, self).__init__( )        
        self.path     = os.path.expanduser( path )
        self.folders_images  = folders_images
        self.metadata        = metadata
        self.data            = []
        self.train           = train
        
        self.data = make_dataset( self.path, self.metadata, self.train )
        
    def __len__(self):
        return len(self.data)
        
    def getname(self, i):
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
        return self.data['Id'][i]        

    def __getitem__(self, i):                
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i        
        if self.train:           
            image_id = self.data['Id'][i]
            prob = self.data['Target'][i]
            image_grby = open_grby(  os.path.join(self.path, self.folders_images ), image_id )
            return image_id, image_grby, prob
        else:
            image_id = self.data['Id'][i]
            image_grby = open_grby( os.path.join(self.path, self.folders_images ) , image_id )
            return image_id, image_grby, 0

        

def test():     

    path = "../input"
    metadata='train.csv' # train.csv, sample_submission.csv
    folders_images='train' #train, test
    train=True #True, False
    dataset = ATLASProvide.create(path=path, train=train, folders_images=folders_images, metadata=metadata )
    iD,image, prob = dataset[1]

    print( len(dataset) )     
    print( iD )
    print( prob )

    #plt.figure( figsize=(8,8) )
    #plt.imshow( image[:,:,:3] )
    #plt.axis('off')
    #plt.show()

# test()