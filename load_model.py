# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:48:54 2019

@author: ISYSRG.COM
"""

from model import *
from data import *
from matplotlib import pyplot as plt
import pickle
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

epochs =1000
batch_size = 64
step_per_epochs = batch_size
validation_steps = batch_size

train_path = '1kelas/train'
test_path = '1kelas/test'
test_path_image = '1kelas/test/image/*.jpg'
test_path_label = '1kelas/test/label/'
result_path = '1kelas/test/test result/'
model_name = 'unet_1kelas_epoch_temp_{}_batch_{}.hdf5'.format(epochs,step_per_epochs)
history_name = 'unet_1kelas_history.pickle'
#pretrain_model_name = 'unet_1_classes_asd2_pretrained_epoch_{}_batch_{}.hdf5'.format(epochs,batch_size)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

#model = get_fractalunet()
model = unet(model_name)

model_checkpoint = ModelCheckpoint(model_name, monitor='loss',verbose=1, save_best_only=True)


import glob
import os
test_path = glob.glob(test_path_image)
results_filename = os.listdir(test_path_label)


model = load_model('unet_1kelas_epoch_temp_1000_batch_64.hdf5'.format(epochs,batch_size), compile=False)
testGene = testGenerator(test_path)
results = model.predict_generator(testGene, len(test_path),verbose=1)
saveResult(result_path, results_filename, results)

