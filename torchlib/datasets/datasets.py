


# dataloader 
from dataprovide import ATLASProvide
from pytvision.transforms.aumentation import ObjectImageTransform

train = 'train'
validation = 'train'
test  = 'train'

class ATLASDataset(object):
    '''
    Management for Human Protein Atlas dataset
    '''

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
                
        obj = ObjectImageTransform( image )
        if self.transform: 
            obj = self.transform( obj )
        image = obj.to_value()
        
        return iD, image, prob 
    

def test():
    path = "../input" #path competition 
    metadata='train.csv' # train.csv, sample_submission.csv
    folders_images='train' #train, test
    train=True #True, False
    dataset = ATLASDataset(path=path, train=train, folders_images=folders_images, metadata=metadata )
    iD,image, prob = dataset[0]

    print( len(dataset) )     
    print( iD )
    print( prob )

    #plt.figure( figsize=(8,8) )
    #plt.imshow( image[:,:,:3] )
    #plt.axis('off')
    #plt.show()



