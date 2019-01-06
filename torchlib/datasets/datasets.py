


import numpy as np
from .dataprovide import ATLASProvide
from .dataprovide import ExternalArtemtprvProvide
from pytvision.transforms.aumentation import ObjectImageTransform
from pytvision.datasets import utility 

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm  import tqdm

import matplotlib.pyplot as plt


train = 'train'
validation = 'train'
test  = 'train'

class ATLASDataset(object):
    '''
    Management for Human Protein Atlas dataset
    '''

    idx_to_class = {
        0:  "Nucleoplasm",  
        1:  "Nuclear membrane",   
        2:  "Nucleoli",   
        3:  "Nucleoli fibrillar center",   
        4:  "Nuclear speckles",
        5:  "Nuclear bodies",   
        6:  "Endoplasmic reticulum",   
        7:  "Golgi apparatus",   
        8:  "Peroxisomes",   
        9:  "Endosomes",   
        10:  "Lysosomes",   
        11:  "Intermediate filaments",   
        12:  "Actin filaments",   
        13:  "Focal adhesion sites",   
        14:  "Microtubules",   
        15:  "Microtubule ends",   
        16:  "Cytokinetic bridge",   
        17:  "Mitotic spindle",   
        18:  "Microtubule organizing center",   
        19:  "Centrosome",   
        20:  "Lipid droplets",   
        21:  "Plasma membrane",   
        22:  "Cell junctions",   
        23:  "Mitochondria",   
        24:  "Aggresome",   
        25:  "Cytosol",   
        26:  "Cytoplasmic bodies",   
        27:  "Rods & rings"
    }

    def __init__(self, 
        path,   
        train=True,
        folders_images='train',
        metadata = 'train.csv',
        ext='png',
        transform=None,
        count=None, 
        num_channels=4,
        ):
        """Initialization       
        """            
           
        self.data = ATLASProvide.create( 
                path, 
                train,
                folders_images, 
                metadata,
                )
        
        self.transform = transform  
        self.count = count if count is not None else len(self.data)   
        self.num_channels = num_channels

    def __len__(self):
        return self.count
    
    def getname(self, idx):
        idx = idx % len(self.data)
        return self.data.getname(idx)

    def __getitem__(self, idx):   
        idx = idx % len(self.data)
        iD, image, prob = self.data[idx]
        #image = (image[:,:,:3 ]*255).astype( np.uint8 )
        #image = (image[:,:, 0 ]*255).astype( np.uint8 )
        image = np.stack( (  image[:,:,0], image[:,:,1]/2 + image[:,:,3]/2, image[:,:,2]/2 + image[:,:,3]/2  ), axis=-1 )
        #image = utility.to_channels(image, 3)
        image = (image*255).astype( np.uint8 )
        
        #print(image.shape, flush=True )
        #print(image.min(), image.max(), flush=True )
        #assert(False)
            
        obj = ObjectImageTransform( image )
        if self.transform: 
            obj = self.transform( obj )
        image = obj.to_value()
        
        #print(image.shape, flush=True )
        #assert(False)
        
        return iD, image, prob 
    

    
class ATLASExtDataset(object):
    '''
    Management for Human Protein Atlas dataset
    Example:
    train_data = ATLASExtDataset(        
         path=args.data, 
         train=True,
         folders_images='train', 
         metadata='train.csv',
         folders_images_external='train_external', 
         metadata_external='train_external.csv',
         count=50000,
         num_channels=args.channels,
         transform=get_transforms_aug( network.size_input ),
         )    
    '''

    idx_to_class = {
        0:  "Nucleoplasm",  
        1:  "Nuclear membrane",   
        2:  "Nucleoli",   
        3:  "Nucleoli fibrillar center",   
        4:  "Nuclear speckles",
        5:  "Nuclear bodies",   
        6:  "Endoplasmic reticulum",   
        7:  "Golgi apparatus",   
        8:  "Peroxisomes",   
        9:  "Endosomes",   
        10:  "Lysosomes",   
        11:  "Intermediate filaments",   
        12:  "Actin filaments",   
        13:  "Focal adhesion sites",   
        14:  "Microtubules",   
        15:  "Microtubule ends",   
        16:  "Cytokinetic bridge",   
        17:  "Mitotic spindle",   
        18:  "Microtubule organizing center",   
        19:  "Centrosome",   
        20:  "Lipid droplets",   
        21:  "Plasma membrane",   
        22:  "Cell junctions",   
        23:  "Mitochondria",   
        24:  "Aggresome",   
        25:  "Cytosol",   
        26:  "Cytoplasmic bodies",   
        27:  "Rods & rings"
    }

    def __init__(self, 
        path,   
        train=True,
        folders_images='train',
        metadata = 'train.csv',
        folders_images_external='train_external',
        metadata_external = 'train_external.csv',
        ext='png',
        transform=None,
        num_channels=4,
        ):
        """Initialization       
        """            
           
        self.data = ATLASProvide.create( 
                path, 
                train,
                folders_images, 
                metadata,
                )        
        self.data_external = ATLASProvide.create( 
                path, 
                train,
                folders_images_external, 
                metadata_external,
                )        
        
        self.count = len(self.data) + len(self.data_external)
        self.index = np.concatenate( ( np.arange( len(self.data) ), np.arange( len(self.data_external) ) ) )     
        self.transform = transform     
        self.num_channels = num_channels
        
    def __len__(self):
        return self.count
    

    def __getitem__(self, i):   
        idx = self.index[i]
        iD, image, prob = self.data[idx] if i < len(self.data) else self.data_external[idx] 
                              
        #image = (image[:,:,:3 ]*255).astype( np.uint8 )
        #image = (image[:,:, 0 ]*255).astype( np.uint8 )
        image = np.stack( (  image[:,:,0], image[:,:,1]/2 + image[:,:,3]/2, image[:,:,2]/2 + image[:,:,3]/2  ), axis=-1 )
        #image = utility.to_channels(image, 3)
        image = (image*255).astype( np.uint8 )
        
        #print(image.shape, flush=True )
        #print(image.min(), image.max(), flush=True )
        #assert(False)
            
        obj = ObjectImageTransform( image )
        if self.transform: 
            obj = self.transform( obj )
        image = obj.to_value()
        
        #print(image.shape, flush=True )
        #assert(False)
        
        return iD, image, prob 
    


class ExternalArtemtprvDataset(object):
    '''
    Management for Human Protein Atlas dataset
    '''

    idx_to_class = {
        0:  "Nucleoplasm",  
        1:  "Nuclear membrane",   
        2:  "Nucleoli",   
        3:  "Nucleoli fibrillar center",   
        4:  "Nuclear speckles",
        5:  "Nuclear bodies",   
        6:  "Endoplasmic reticulum",   
        7:  "Golgi apparatus",   
        8:  "Peroxisomes",   
        9:  "Endosomes",   
        10:  "Lysosomes",   
        11:  "Intermediate filaments",   
        12:  "Actin filaments",   
        13:  "Focal adhesion sites",   
        14:  "Microtubules",   
        15:  "Microtubule ends",   
        16:  "Cytokinetic bridge",   
        17:  "Mitotic spindle",   
        18:  "Microtubule organizing center",   
        19:  "Centrosome",   
        20:  "Lipid droplets",   
        21:  "Plasma membrane",   
        22:  "Cell junctions",   
        23:  "Mitochondria",   
        24:  "Aggresome",   
        25:  "Cytosol",   
        26:  "Cytoplasmic bodies",   
        27:  "Rods & rings"
    }

    def __init__(self, 
        path,   
        train=True,
        folders_images='external_data',
        metadata = 'train.csv',
        ext='png',
        transform=None,
        count=None, 
        num_channels=4,
        ):
        """Initialization       
        """            
           
        self.data = ExternalArtemtprvProvide.create( 
                path, 
                train,
                folders_images, 
                metadata,
                )
        
        self.transform = transform  
        self.count = count if count is not None else len(self.data)   
        self.num_channels = num_channels

    def __len__(self):
        return self.count
    
    def getname(self, idx):
        idx = idx % len(self.data)
        return self.data.getname(idx)

    def __getitem__(self, idx):   
        idx = idx % len(self.data)
        iD, image, prob = self.data[idx]
        #image = (image[:,:,:3 ]*255).astype( np.uint8 )
        #image = (image[:,:, 0 ]*255).astype( np.uint8 )
        #image = np.stack( (  image[:,:,0], image[:,:,1]/2 + image[:,:,3]/2, image[:,:,2]/2 + image[:,:,3]/2  ), axis=-1 )
        #image = np.stack( (image), axis=-1 )
        image=image.reshape((image.shape[1],image.shape[2],image.shape[3]))
        #print("TESTANDO SHAPE")
        #print(image.shape)
        #image = np.stack( (image[:,:,0],image[:,:,1],image[:,:,2]) )
        #print(image.shape)
        #print("--------------------")


        #image = utility.to_channels(image, 3)
        image = (image*255).astype( np.uint8 )
        
        #print(image.shape, flush=True )
        #print(image.min(), image.max(), flush=True )
        #assert(False)
        

        obj = ObjectImageTransform( image )
        if self.transform: 
            obj = self.transform( obj )
        image = obj.to_value()
        
        print(image.shape, flush=True )
        #assert(False)
        
        return iD, image, prob 




def test():
    path = "/home/rapela/Downloads/kaggle-human-protein-atlas/external_data_artemtprv/external-data-for-protein-atlas/" #path competition 
    metadata='train.csv' # train.csv, sample_submission.csv
    folders_images='external_data' #train, test
    train=True #True, False
    dataset = ExternalArtemtprvDataset(path=path, train=train, folders_images=folders_images, metadata=metadata )
    iD,image, prob = dataset[0]

    print( len(dataset) )     
    print( iD )
    print( prob )


    print(image.shape)

    #plt.figure( figsize=(8,8) )
    #plt.imshow( image )
    #plt.axis('off')
    #plt.show()

 
    
    ## Debug img open
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    


if __name__ == '__main__':
    test()