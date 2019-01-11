#!/bin/bash

# parameters
DATA=$HOME/.kaggle/competitions/human-protein-atlas-image-classification
NAMEDATASET='atlas'
PROJECT='../out/netruns'
EPOCHS=1000
BATCHSIZE=128 #128, 32
LEARNING_RATE=0.0001  # 0.1, 0.03, 0001, 00001
MOMENTUM=0.9
PRINT_FREQ=100
WORKERS=32 #10
RESUME='chk000255.pth.tar' #model_best, chk000000
GPU=0
ARCH='resnet18' # resnet18, resnet34, resnet50, preactresnet18, inception_v4, nasnetalarge
LOSS='mix' #focal, bcewl, mix
OPT='adam' #sgd, adam
SCHEDULER='step'
SNAPSHOT=5
NUMCLASS=28
NUMCHANNELS=3 #4
IMAGESIZE=512 #1024, 512, 331
EXP_NAME='atlas_baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_005' #11


# rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
# rm -rf $PROJECT/$EXP_NAME/
# mkdir $PROJECT
# mkdir $PROJECT/$EXP_NAME


## execute
python ../train.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--batch-size=$BATCHSIZE \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--print-freq=$PRINT_FREQ \
--workers=$WORKERS \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER \
--arch=$ARCH \
--num-classes=$NUMCLASS \
--name-dataset=$NAMEDATASET \
--channels=$NUMCHANNELS \
--image-size=$IMAGESIZE \
--parallel \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \
