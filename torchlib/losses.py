

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def to_gpu( x, cuda ):
    return x.cuda() if cuda else x

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels.long()]     # [N,D]

def flatten(x):        
    x_flat = x.clone()
    x_flat = x_flat.view(x.shape[0], -1)
    return x_flat



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25,gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        t = Variable(y).cuda()  # [N,20]
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = self.alpha*t + (1-self.alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)


# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
# https://arxiv.org/pdf/1708.02002.pdf
class FocalLossV1(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss        
        return loss.sum(dim=1).mean()

    
class FocalLossV2(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma 
        
    def forward(self, y_pred, y_true):           
        y_pred_log =  F.logsigmoid(y_pred)
        weight = (1 - F.sigmoid(y_pred) ) ** self.gamma        
        logpy = torch.sum( weight * y_pred_log * y_true, dim=1 )
        loss  = -torch.mean(logpy)
        return loss
        



# credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
#def f1(y_true, y_pred):
#    #y_pred = K.round(y_pred)
#    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
#    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
#    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
#    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
#    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
#
#    p = tp / (tp + fp + K.epsilon())
#    r = tp / (tp + fn + K.epsilon())
#
#    f1 = 2*p*r / (p+r+K.epsilon())
#    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#    return K.mean(f1)

#def f1_loss(y_true, y_pred):   
#    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
#    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
#    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
#    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
#    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
#
#    p = tp / (tp + fp + K.epsilon())
#    r = tp / (tp + fn + K.epsilon())
#
#    f1 = 2*p*r / (p+r+K.epsilon())
#    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#    return 1-K.mean(f1)

class F1Loss(nn.Module):    
    def __init__(self, eps=1e-6, beta=2):
        super(F1Loss, self).__init__()
        self.eps = eps 
        self.beta = beta        
    def forward(self, logits, labels  ):
        eps = self.eps
        beta = self.beta
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + eps
        num_pos_hat = torch.sum(l, 1) + eps
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall  = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + eps)
        loss = fs.sum() / batch_size
        return (1 - loss)


class DiceLoss(nn.Module):    
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, y_pred, y_true, weight=None ):
        y_pred = F.sigmoid(y_pred)
        smooth = 1.
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        score = (2. * torch.sum(y_true_f * y_pred_f) + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return 1. - score

    
class MixLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.0):
        super(MixLoss, self).__init__()
        self.loss_mce = nn.BCEWithLogitsLoss( size_average=True )
        self.loss_dice = DiceLoss()    
        self.loss_f1 = F1Loss()
                
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true ):
        alpha = self.alpha
        gamma = self.gamma
        loss_m  = self.loss_mce( y_pred, y_true)    
        loss_d  = self.loss_dice( y_pred, y_true )     
        
        isel = [8,9,10,15,16,17,20,24,26,27] #8,9,10,15,16,17,20,24,26,27 | 8,9,10,15,20,27
        for i in isel:
            loss_d  += self.loss_dice( y_pred[:,i] , y_true[:,i] ) 
        loss_d = loss_d/(len(isel) - 1)
        
        loss_f  = self.loss_f1( y_pred, y_true )         
        loss = alpha*loss_m + gamma*loss_d + loss_f
        return loss
    
    
# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
class MultAccuracyV1(nn.Module):    
    def __init__(self, th=0.5 ):
        super(MultAccuracyV1, self).__init__()
        self.th=th

    def forward(self, yhat, y):
        """Computes the precision@k for the specified values of k"""       
        
        yhat = F.sigmoid(yhat)
        yhat = (yhat > self.th ).int()
        y = y.int()
        return (yhat==y).float().mean()

    
class F_score(nn.Module):    
    def __init__(self, threshold=0.5, beta=2 ):
        super(F_score, self).__init__()
        self.threshold=threshold
        self.beta = beta

    def forward(self, logit, label):

        threshold = self.threshold
        beta = self.beta
        
        prob = torch.sigmoid(logit)
        prob = prob > threshold
        label = label > threshold

        TP = (prob & label).sum(1).float()
        TN = ((~prob) & (~label)).sum(1).float()
        FP = (prob & (~label)).sum(1).float()
        FN = ((~prob) & label).sum(1).float()

        precision = torch.mean(TP / (TP + FP + 1e-12))
        recall = torch.mean(TP / (TP + FN + 1e-12))
        F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
        
        return F2.mean(0)




## Baseline clasification
class TopkAccuracy(nn.Module):
    
    def __init__(self, topk=(1,)):
        super(TopkAccuracy, self).__init__()
        self.topk = topk

    def forward(self, output, target):
        """Computes the precision@k for the specified values of k"""
        
        maxk = max(self.topk)
        batch_size = target.size(0)

        pred = output.topk(maxk, 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append( correct_k.mul_(100.0 / batch_size) )

        return res



class ConfusionMeter( object ):
    """Maintains a confusion matrix for a given calssification problem.
    https://github.com/pytorch/tnt/tree/master/torchnet/meter

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """

        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf



