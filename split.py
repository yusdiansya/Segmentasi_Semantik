from sklearn.model_selection import train_test_split
import glob
import numpy as np
import os
import shutil

data_type = "sebelum split"
idx = 0



root_folder_image = "C:/Users/Windows 10/Desktop/SK6A/Jaringan Syaraf Tiruan/UAS JST/Hasil/image/*.jpg".format(data_type, idx)
root_folder_label = "C:/Users/Windows 10/Desktop/SK6A/Jaringan Syaraf Tiruan/UAS JST/Hasil/label/*.jpg".format(data_type,idx)

image = glob.glob(root_folder_image)
label = glob.glob(root_folder_label)

image = np.array(image)
label = np.array(label)

index_image = np.arange(len(image))
index_label = np.arange(len(label))

idx_image_train , idx_image_test, _ , _ = train_test_split(index_image, index_image, test_size  = 0.2, random_state = 64)

image_train = image[idx_image_train]
image_test = image[idx_image_test]




filenames =glob.glob("C:/Users/Windows 10/Desktop/SK6A/Jaringan Syaraf Tiruan/UAS JST/Hasil/label/*.jpg")

## root path yang digunakan
root_path_saving_train_image = 'train split/image'  ## dimana gambar akan disave
root_path_saving_test_image = 'test split/image'  ## dimana gambar akan disave
root_path_saving_train_label = 'train split/label'  ## dimana gambar akan disave
root_path_saving_test_label = 'test split/label'  ## dimana gambar akan disave

#make new directory
os.makedirs(root_path_saving_train_image, exist_ok = True)
os.makedirs(root_path_saving_test_image, exist_ok = True)
os.makedirs(root_path_saving_train_label, exist_ok = True)
os.makedirs(root_path_saving_test_label, exist_ok = True)


## split data train image

for idx, path in enumerate(image_train):
    filename = path.split('\\')[-1]
    shutil.copy2(path,root_path_saving_train_image.format(filename))

 
path_test = "C:/Users/Windows 10/Desktop/SK6A/Jaringan Syaraf Tiruan/UAS JST/Hasil/label/"
for i in range (len(image)):
    for j in range(len(image_train)):
        if(image[i]==image_train[j]):
            filename_train = image[i].split('\\')[-1]
            shutil.copy2(path_test+filename_train,root_path_saving_train_label.format(filename_train))
  
    
for idx, path in enumerate(image_test):
    filename = path.split('\\')[-1]
    shutil.copy2(path,root_path_saving_test_image.format(filename))
    
    
path_test = "C:/Users/Windows 10/Desktop/SK6A/Jaringan Syaraf Tiruan/UAS JST/Hasil/label/"
for i in range (len(image)):
    for j in range(len(image_test)):
        if(image[i]==image_test[j]):
            filename_train = image[i].split('\\')[-1]
            shutil.copy2(path_test+filename_train,root_path_saving_test_label.format(filename_train))
      
