# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:28:29 2019

@author: ISYSRG.COM
"""

from model import *
from data import *
from matplotlib import pyplot as plt
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


epochs = 1000
batch_size = 64


train_path = '1 Class Normal/train'
test_path = '1 Class Normal/test/'
test_path_2 = '1 Class Normal/test/label/*.jpg'
result_path = '1 Class Normal/test/test result'
model_name = 'unet_1_classes_normal_epoch_temp_{}_batch_{}.hdf5'.format(epochs,batch_size)
#pretrain_model_name = 'unet_1_classes_asd2_pretrained_epoch_{}_batch_{}.hdf5'.format(epochs,batch_size)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,train_path,'image','label',data_gen_args,save_to_dir = None)
testGene = valGenerator(2,test_path,'image','label',save_to_dir = None)

model = unet()
#model = unet(model_name)

model_checkpoint = ModelCheckpoint(model_name, monitor='loss',verbose=1, save_best_only=True)
#model_checkpoint = ModelCheckpoint(pretrain_model_name, monitor='loss',verbose=1, save_best_only=True)

history_train = model.fit_generator(myGene,steps_per_epoch = batch_size, epochs=epochs,callbacks=[model_checkpoint], validation_data=testGene, validation_steps = batch_size)

fig,(ax0) = plt.subplots(nrows=1, figsize=(12,6))
ax0.plot(history_train.history['acc'],'red', label='Akurasi Training')
ax0.plot(history_train.history['val_acc'], 'blue', label='Akurasi Testing')
ax0.plot(label='Accuracy', loc='upper left')
ax0.set_title('Model Accuracy')
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Accuracy")
ax0.legend()
plt.savefig('Model_Akurasi_NORMAL.png')
plt.close()

fig,(ax1) = plt.subplots(nrows=1, figsize=(12,6))
ax1.plot(history_train.history['loss'],'red', label='Loss Training')
ax1.plot(history_train.history['val_loss'], 'blue', label='Loss Testing')
ax1.plot(label='Loss', loc='upper left')
ax1.set_title('Model Loss')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
plt.savefig('Model_Loss_NORMAL.png')
plt.close()

import pickle
with open('history_normal.picle', 'wb') as fsave:
    pickle.dump(history_train, fsave)

# =============================================================================
# import glob
# import os
# test_path = glob.glob(test_path)
# results_filename = os.listdir(test_path_2)
# 
# #model = load_model('unet_2_classes_epoch_{}_batch_{}.hdf5'.format(epochs,batch_size))
# testGene = testGenerator(test_path)
# results = model.predict_generator(testGene, len(test_path),verbose=1)
# saveResult(result_path, results_filename, results)
# =============================================================================
