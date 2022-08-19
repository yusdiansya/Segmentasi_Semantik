import pickle
from matplotlib import pyplot as plt
history_train = pickle.load(open('unet_2kelas_history.pickle','rb'))

fig,(ax0) = plt.subplots(nrows=1, figsize=(12,6))
ax0.plot(history_train.history['accuracy'],'red', label='Training Accuracy')
ax0.plot(history_train.history['val_acc'], 'blue', label='Testing Accuracy')
ax0.plot(label='Accuracy', loc='upper left')
ax0.set_title('Model Accuracy')
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Accuracy")
ax0.legend()
plt.savefig('Model_Accuracy_2kelas.png')
plt.close()