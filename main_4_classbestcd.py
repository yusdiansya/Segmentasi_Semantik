# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:48:54 2019

@author: ISYSRG.COM
"""
from datetime import datetime
start_time = datetime.now()

from model import *
from data import *
from matplotlib import pyplot as plt
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


epochs = 1000
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

myGene = trainGenerator(10,train_path,'image','label',data_gen_args,save_to_dir = None)
testGene = valGenerator(10,test_path,'image','label')

model = unet()
#model = unet(model_name)

model_checkpoint = ModelCheckpoint(model_name, monitor='loss',verbose=1, save_best_only=True)
#model_checkpoint = ModelCheckpoint(pretrain_model_name, monitor='loss',verbose=1, save_best_only=True)

history_train = model.fit_generator(myGene,steps_per_epoch = batch_size, epochs=epochs,callbacks=[model_checkpoint], validation_steps=validation_steps,validation_data=testGene)


with open(history_name, 'wb') as fsave:
    pickle.dump(history_name,fsave)
    
fig,(ax0) = plt.subplots(nrows=1, figsize=(12,6))
ax0.plot(history_train.history['accuracy'],'red', label='Training Accuracy')
ax0.plot(history_train.history['val_accuracy'], 'blue', label='Testing Accuracy')
ax0.plot(label='Accuracy', loc='upper left')
ax0.set_title('Model Accuracy')
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Accuracy")
ax0.legend()
plt.savefig('Model_Accuracy_1kelas.png')
plt.close()

fig,(ax1) = plt.subplots(nrows=1, figsize=(12,6))
ax1.plot(history_train.history['loss'],'red', label='Training Loss')
ax1.plot(history_train.history['val_loss'], 'blue', label='Testing Loss')
ax1.plot(label='Loss', loc='upper left')
ax1.set_title('Model Loss')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
plt.savefig('Model_Loss1kelas.png')
plt.close()

import glob
import os
test_path = glob.glob(test_path_image)
results_filename = os.listdir(test_path_label)

#model = load_model('unet_2_classes_epoch_{}_batch_{}.hdf5'.format(epochs,batch_size))
testGene = testGenerator(test_path)
results = model.predict_generator(testGene, len(test_path),verbose=1)
saveResult(result_path, results_filename, results)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))