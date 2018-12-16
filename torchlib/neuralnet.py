
# STD MODULES 
import os
import math
import shutil
import time
import numpy as np
from tqdm import tqdm

# TORCH MODULE
import torch
import torch.nn as nn
import torch.nn.functional as F

# PYTVISION MODULE
from pytvision.neuralnet import NeuralNetAbstract
from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
from pytvision import utils as pytutils
from pytvision import graphic as gph
from pytvision import netlearningrate

# LOCAL MODULE
from . import models as nnmodels
from . import losses as nloss
from . import utils  as ult



class NeuralNetClassifier(NeuralNetAbstract):
    r"""Convolutional Neural Net for classification
    Args:
        patchproject (str): path project
        nameproject (str):  name project
        no_cuda (bool): system cuda (default is True)
        parallel (bool)
        seed (int)
        print_freq (int)
        gpu (int)
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0
        ):
        super(NeuralNetClassifier, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )

 
    def create(self, 
        arch, 
        num_output_channels, 
        num_input_channels,  
        loss, 
        lr,          
        optimizer, 
        lrsch,  
        momentum=0.9,
        weight_decay=5e-4,        
        pretrained=False,
        topk=(1,),
        size_input=128,
        ):
        """
        Create
        Args:
            arch (string): architecture
            num_output_channels, 
            num_input_channels,  
            loss (string):
            lr (float): learning rate
            momentum,
            optimizer (string) : 
            lrsch (string): scheduler learning rate
            pretrained (bool)
        """

        cfg_opt= { 'momentum':0.9, 'weight_decay':5e-4 } 
        cfg_scheduler= { 'step_size':100, 'gamma':0.1  }
                    
        super(NeuralNetClassifier, self).create( 
            arch, 
            num_output_channels, 
            num_input_channels, 
            loss, 
            lr, 
            optimizer, 
            lrsch, 
            pretrained, 
            cfg_opt=cfg_opt,
            cfg_scheduler=cfg_scheduler,
        )
        
        self.size_input = size_input
        self.accuracy = nloss.TopkAccuracy( topk ) ### <---- !!!DEFINE ACURATE!!!!!

        #self.cnf = nloss.ConfusionMeter( self.num_output_channels, normalized=True )
        #self.visheatmap = gph.HeatMapVisdom( env_name=self.nameproject )

        # Set the graphic visualization
        self.logger_train = Logger( 'Trn', ['loss'], ['acc'], self.plotter  )
        self.logger_val   = Logger( 'Val', ['loss'], ['acc'], self.plotter )
              

    
    def training(self, data_loader, epoch=0):

        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, ( iD, image, prob  ) in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            x, y = image, prob  
            batch_size = x.size(0)

            if self.cuda:
                x = x.cuda() 
                y = y.cuda()             

            # fit (forward)
            yhat = self.net(x)

            # measure accuracy and record loss
            loss = self.criterion( yhat, y  )            
            pred = self.accuracy( yhat.data, y )
              
            # optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                      
            # update
            self.logger_train.update(
                {'loss': loss.data[0] },
                {'acc': pred.data[0] },
                batch_size,
                )
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )
    

    def evaluate(self, data_loader, epoch=0):
        
        self.logger_val.reset()
        #self.cnf.reset()
        batch_time = AverageMeter()
        

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (iD, image, prob) in enumerate(data_loader):
                
                # get data (image, label)
                x, y = image, prob #.argmax(1).long()
                batch_size = x.size(0)

                if self.cuda:
                    x = x.cuda() 
                    y = y.cuda() 

                
                # fit (forward)
                yhat = self.net(x)

                # measure accuracy and record loss
                loss = self.criterion(yhat, y )      
                pred = self.accuracy(yhat.data, y.data ) 

                #self.cnf.add( outputs.argmax(1), y ) 

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update(
                {'loss': loss.data[0] },
                {'acc': loss.data[0] },
                batch_size,
                )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['acc']['acc'].avg
        
        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )
        
        #print('Confusion Matriz')
        #print(self.cnf.value(), flush=True)
        #print('\n')
        #self.visheatmap.show('Confusion Matriz', self.cnf.value())                
        
        return acc
    
        
    def predict(self, data_loader):
        Yhat = []
        iDs  = []
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (iD, image, prob) in enumerate( tqdm(data_loader) ):
                x = image.cuda() if self.cuda else image                                    
                yhat = self.net(x).cpu().numpy()
                Yhat.append( yhat )
                iDs.append( iD )
        Yhat = np.stack( Yhat, axis=0 )        
        iDs = np.stack( iDs, axis=0 )    
        return iDs, Yhat
      
    def __call__(self, image):        
        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image 
            yhat = self.net(x).cpu().numpy()
        return yhat

  
    
    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained):
        """
        Create model
            @arch (string): select architecture
            @num_classes (int)
            @num_channels (int)
            @pretrained (bool)
        """    

        self.net = None
        self.size_input = 0     
        
        kw = {'num_classes': num_output_channels, 'num_channels': num_input_channels, 'pretrained': pretrained}
        self.net = nnmodels.__dict__[arch](**kw)
        
        self.s_arch = arch
        self.size_input = self.net.size_input        
        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels

        if self.cuda == True:
            self.net.cuda()
        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids= range( torch.cuda.device_count() ))

    def _create_loss(self, loss):

        # create loss
        if loss == 'cross':
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif loss == 'mse':
            self.criterion = nn.MSELoss(size_average=True).cuda()
        elif loss == 'l1':
            self.criterion = nn.L1Loss(size_average=True).cuda()
        else:
            assert(False)

        self.s_loss = loss


