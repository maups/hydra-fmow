"""
Copyright 2017 The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = 'jhuapl'
__version__ = 0.1

import json
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications import imagenet_utils
from data_ml_functions.mlFunctions import get_cnn_model,img_metadata_generator,get_lstm_model,codes_metadata_generator
from data_ml_functions.dataFunctions import prepare_data,calculate_class_weights
from sklearn.utils import class_weight
import numpy as np
import os

from data_ml_functions.multi_gpu import make_parallel

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import time

class FMOWBaseline:
    def __init__(self, params=None, parser=None):
        """
        Initialize baseline class, prepare data, and calculate class weights.
        :param params: global parameters, used to find location of the dataset and json file
        :return: 
        """
        self.params = params
       
        if (parser['train']):
           self.params.train_cnn = True
        if (parser['path']):
           self.params.path = parser['path']
        if (parser['prepare']):
            prepare_data(params)
        if (parser['algorithm']):
           self.params.algorithm = parser['algorithm']
        if (parser['nm']):
           self.params.use_metadata = False 
        if (parser['test']):
           self.params.test_cnn = True
        if (parser['num_gpus']):
           self.params.num_gpus = parser['num_gpus']
        if (parser['load_weights']):
           self.params.model_weights = parser['load_weights']
        if (parser['num_epochs']):
           self.params.cnn_epochs = parser['num_epochs']
        if (parser['batch_size']):
           self.params.batch_size_cnn = parser['batch_size'] 
        if (parser['fine_tunning']):
           self.params.fine_tunning = True
        if (parser['class_weights']):
           self.params.class_weights = parser['class_weights'] 
        if (parser['prefix']):
           self.params.prefix = parser['prefix'] 
        if (parser['generator']):
           self.params.generator = parser['generator']
        if (parser['database']):
           self.params.database = parser['database']

        if self.params.use_metadata:
            self.params.files['cnn_model'] = os.path.join(self.params.directories['cnn_models'], 'cnn_model_with_metadata.model')
            self.params.files['lstm_model'] = os.path.join(self.params.directories['lstm_models'], 'lstm_model_with_metadata.model')
            self.params.files['cnn_codes_stats'] = os.path.join(self.params.directories['working'], 'cnn_codes_stats_with_metadata.json')
            self.params.files['lstm_training_struct'] = os.path.join(self.params.directories['working'], 'lstm_training_struct_with_metadata.json')
            self.params.files['lstm_test_struct'] = os.path.join(self.params.directories['working'], 'lstm_test_struct_with_metadata.json')
        else:
            self.params.files['cnn_model'] = os.path.join(self.params.directories['cnn_models'], 'cnn_model_no_metadata.model')
            self.params.files['lstm_model'] = os.path.join(self.params.directories['lstm_models'], 'lstm_model_no_metadata.model')
            self.params.files['cnn_codes_stats'] = os.path.join(self.params.directories['working'], 'cnn_codes_stats_no_metadata.json')
            self.params.files['lstm_training_struct'] = os.path.join(self.params.directories['working'], 'lstm_training_struct_no_metadata.json')
            self.params.files['lstm_test_struct'] = os.path.join(self.params.directories['working'], 'lstm_test_struct_no_metadata.json')
    
    def train_cnn(self):
        """
        Train CNN with or without metadata depending on setting of 'use_metadata' in params.py.
        :param: 
        :return: 
        """
        
        if (self.params.database == 'v1'):
           print ('self.params.database = v1')
           trainData = json.load(open(self.params.files['training_struct1']))
           metadataStats = json.load(open(self.params.files['dataset_stats1']))
        elif (self.params.database == 'v2'):
           print ('self.params.database = v2')
           trainData = json.load(open(self.params.files['training_struct2']))
           metadataStats = json.load(open(self.params.files['dataset_stats2']))
        elif (self.params.database == 'v3'):
           print ('self.params.database = v3')
           trainData = json.load(open(self.params.files['training_struct3']))
           metadataStats = json.load(open(self.params.files['dataset_stats3']))
        else:
           print ('Error: define a dataset!')

        model = get_cnn_model(self.params, self.params.algorithm)
        model = make_parallel(model, self.params.num_gpus)
        if (self.params.model_weights != ''):
           model.load_weights(self.params.model_weights, by_name=True)
        model.compile(optimizer=Adam(lr=self.params.cnn_adam_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        if (self.params.class_weights == 'no_weights'):
           print ('self.params.class_weights = no_weights')
           classWeights = {0:1.0, 1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 6:1.0, 7:1.0, 8:1.0, 9:1.0, 10:1.0, 11:1.0, 12:1.0, 13:1.0, 14:1.0, 15:1.0, 16:1.0, 17:1.0, 18:1.0, 19:1.0, 20:1.0, 21:1.0, 22:1.0, 23:1.0, 24:1.0, 25:1.0, 26:1.0, 27:1.0, 28:1.0, 29:1.0, 30:1.0, 31:1.0, 32:1.0, 33:1.0, 34:1.0, 35:1.0, 36:1.0, 37:1.0, 38:1.0, 39:1.0, 40:1.0, 41:1.0, 42:1.0, 43:1.0, 44:1.0, 45:1.0, 46:1.0, 47:1.0, 48:1.0, 49:1.0, 50:1.0, 51:1.0, 52:1.0, 53:1.0, 54:1.0, 55:1.0, 56:1.0, 57:1.0, 58:1.0, 59:1.0, 60:1.0, 61:1.0, 62:1.0}
        elif (self.params.class_weights == 'class_weights'):
           print ('self.params.class_weights = class_weights')
           classWeights = {0:1.0, 1:0.6, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 6:1.0, 7:1.0, 8:1.4, 9:1.0, 10:1.0, 11:1.4, 12:0.6, 13:1.0, 14:0.6, 15:1.4, 16:1.0, 17:1.4, 18:1.4, 19:0.6, 20:1.0, 21:1.4, 22:1.0, 23:1.0, 24:1.0, 25:1.0, 26:1.0, 27:1.0, 28:1.0, 29:0.6, 30:1.0, 31:0.6, 32:1.0, 33:1.0, 34:1.0, 35:1.0, 36:1.0, 37:1.4, 38:1.0, 39:1.0, 40:1.0, 41:1.0, 42:1.0, 43:1.0, 44:1.4, 45:1.0, 46:1.0, 47:1.0, 48:0.6, 49:1.4, 50:0.6, 51:1.0, 52:1.0, 53:1.0, 54:1.0, 55:1.0, 56:1.0, 57:1.4, 58:0.6, 59:1.0, 60:1.0, 61:0.6, 62:1.0}
        elif (self.params.class_weights == 'class_pond'):
           print ('self.params.class_weights = class_pond')
           classWeights = {0:0.65, 1:0.95, 2:0.83, 3:0.84, 4:0.80, 5:0.94, 6:0.91, 7:0.78, 8:0.95, 9:0.85, 10:0.82, 11:0.88, 12:0.30, 13:0.85, 14:0.98, 15:0.58, 16:0.81, 17:0.87, 18:0.82, 19:0.98, 20:0.80, 21:0.81, 22:0.87, 23:0.62, 24:0.85, 25:0.81, 26:0.87, 27:0.97, 28:0.87, 29:0.56, 30:0.73, 31:0.99, 32:0.80, 33:0.84, 34:0.83, 35:0.53, 36:0.40, 37:0.82, 38:0.94, 39:0.85, 40:0.81, 41:0.84, 42:0.20, 43:0.95, 44:0.82, 45:0.91, 46:0.98, 47:0.80, 48:0.63, 49:0.82, 50:0.82, 51:0.99, 52:0.80, 53:0.81, 54:0.82, 55:0.71, 56:0.81, 57:0.74, 58:0.78, 59:0.85, 60:0.81, 61:0.82, 62:0.94}
        elif (self.params.class_weights == 'sklearn_class_weight'):
           print ('self.params.class_weights = sklearn_class_weight')
           train_qtd = [10381, 1660, 5675, 5179, 6715, 1887, 3044, 7079, 1753, 4847, 5999, 3828, 28445, 4946, 2433, 13875, 6486, 4210, 6067, 2399, 6616, 6417, 4447, 12504, 5090, 6144, 4383, 998, 4267, 14729, 8935, 255, 6676, 5165, 5727, 15404, 22198, 6063, 2140, 5064, 6402, 5452, 33114, 1559, 6120, 2848, 758, 6794, 12386, 5738, 5791, 258, 6778, 6338, 5877, 9703, 6228, 8487, 7270, 4917, 6263, 5921, 1862]

           Y = []
           for i in range(len(train_qtd)):
             for j in range(train_qtd[i]):
                Y.append(i)
           print (len(Y))

           classWeights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)

           print (classWeights)

        train_datagen = img_metadata_generator(self.params, trainData, metadataStats)

        def lr_scheduler1 (epoch):
          if self.params.lr_mode is 'progressive_drops':
            if epoch >= 0.75 * self.params.cnn_epochs:
                lr = 1e-6
            elif epoch >= 0.15 * self.params.cnn_epochs:
                lr = 1e-5
            else:
                lr = 1e-4
          print('lr_scheduler1 - epoch: %d, lr: %f' % (epoch, lr))
          return lr

        def lr_scheduler2 (epoch):
          if self.params.lr_mode is 'progressive_drops':
            if epoch > 0.75 * self.params.cnn_epochs:
                lr = 1e-6
            elif epoch > 0.45 * self.params.cnn_epochs:
                lr = 1e-5
            else:
                lr = 1e-4
          print('lr_scheduler2 - epoch: %d, lr: %f' % (epoch, lr))
          return lr

        if (self.params.fine_tunning): 
           lr_decay = LearningRateScheduler(lr_scheduler1)
        else:
           lr_decay = LearningRateScheduler(lr_scheduler2)

        print("training")
        fileName = 'weights.' + self.params.database + '.' + self.params.algorithm + '.' + self.params.prefix + '.{epoch:02d}.hdf5'
        filePath = os.path.join(self.params.directories['cnn_checkpoint_weights'], fileName)
        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=0, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=1)
        
        callbacks_list = [checkpoint,lr_decay]

        model.fit_generator(train_datagen,
                            steps_per_epoch=(len(trainData) / self.params.batch_size_cnn + 1),
                            epochs=self.params.cnn_epochs, class_weight=classWeights, callbacks=callbacks_list)
        
        fileNameEnd = 'weights.final.' + self.params.database + '.' + self.params.algorithm + '.' + self.params.prefix + '.hdf5'
        filePathEnd = os.path.join(self.params.directories['cnn_checkpoint_weights'], fileNameEnd)
        model.save(filePathEnd)

    def test_models(self):

        if (self.params.database == 'v1'):
           metadataStats = json.load(open(self.params.files['dataset_stats1']))
        elif (self.params.database == 'v2'):
           metadataStats = json.load(open(self.params.files['dataset_stats2']))
        elif (self.params.database == 'v3'):
           metadataStats = json.load(open(self.params.files['dataset_stats3']))
        else:
           print ('Error: define a dataset!')
 
        metadataMean = np.array(metadataStats['metadata_mean'])
        metadataMax = np.array(metadataStats['metadata_max'])

        cnnModel = get_cnn_model(self.params, self.params.algorithm)
        cnnModel = make_parallel(cnnModel, self.params.num_gpus)
        cnnModel.load_weights(self.params.model_weights)
        cnnModel = cnnModel.layers[-2]
     
        index = 0
        timestr = time.strftime("%Y%m%d-%H%M%S")
        
        fidCNN1 = open(os.path.join(self.params.directories['predictions'], 'predictions-challenge-%s-%s-%s-clas-cnn-%s.txt' % (self.params.algorithm, self.params.database, self.params.prefix, timestr)), 'w')
        fidCNN2 = open(os.path.join(self.params.directories['predictions'], 'predictions-challenge-%s-%s-%s-vect-cnn-%s.txt' % (self.params.algorithm, self.params.database, self.params.prefix, timestr)), 'w')

        def walkdir(folder):
            for root, dirs, files in os.walk(folder):
                if len(files) > 0:
                    yield (root, dirs, files)
        
        num_sequences = 0
        for _ in walkdir(self.params.directories['test_data']):
            num_sequences += 1

        for root, dirs, files in tqdm(walkdir(self.params.directories['test_data']), total=num_sequences):
            if len(files) > 0:
                imgPaths = []
                metadataPaths = []
                slashes = [i for i,ltr in enumerate(root) if ltr == '/']
                bbID = int(root[slashes[-1]+1:])

            for file in files:
                if (self.params.database == 'v1') and file.endswith('_rgba.jpg'):
                    imgPaths.append(os.path.join(root,file))
                    metadataPaths.append(os.path.join(root, file[:-4]+'_features.json'))
                elif (self.params.database == 'v2') and file.endswith('_rgbb.jpg'):
                    imgPaths.append(os.path.join(root,file))
                    metadataPaths.append(os.path.join(root, file[:-4]+'_features.json'))
                elif (self.params.database == 'v3') and (file.endswith('_rgba.jpg') or file.endswith('_msrgba.jpg')):
                    imgPaths.append(os.path.join(root,file))
                    metadataPaths.append(os.path.join(root, file[:-4]+'_features.json'))
                   
            if len(files) > 0:
                inds = []
                for metadataPath in metadataPaths:
                    underscores = [ind for ind,ltr in enumerate(metadataPath) if ltr == '_']
                    inds.append(int(metadataPath[underscores[-3]+1:underscores[-2]]))
                inds = np.argsort(np.array(inds)).tolist()
                
                currBatchSize = len(inds)
                imgdata = np.zeros((currBatchSize, self.params.target_img_size[0], self.params.target_img_size[1], self.params.num_channels))
                metadataFeatures = np.zeros((currBatchSize, self.params.metadata_length))

                for ind in inds:
                    img = image.load_img(imgPaths[ind])
                    img = image.img_to_array(img)
                    img.setflags(write=True)
                    imgdata[ind,:,:,:] = img

                    features = np.array(json.load(open(metadataPaths[ind])))
                    features = np.divide(features - metadataMean, metadataMax)
                    metadataFeatures[ind,:] = features
                    
                imgdata = imagenet_utils.preprocess_input(imgdata)
                imgdata = imgdata / 255.0
                
                if self.params.use_metadata:
                   predictionsCNN = np.sum(cnnModel.predict([imgdata, metadataFeatures], batch_size=currBatchSize), axis=0)
                else:
                   predictionsCNN = np.sum(cnnModel.predict(imgdata, batch_size=currBatchSize), axis=0)
                
            if len(files) > 0:
                if self.params.test_cnn:
                    predCNN = np.argmax(predictionsCNN)
                    oursCNNStr = self.params.category_names[predCNN]

                    fidCNN1.write('%d;%s;\n' % (bbID,oursCNNStr))
                    
                    fidCNN2.write("%d " % (bbID)),
                    for pred in predictionsCNN:
                       fidCNN2.write("%5.12f " % (pred)),
                    fidCNN2.write("\n")

                index += 1
                
        fidCNN1.close()
        fidCNN2.close()


