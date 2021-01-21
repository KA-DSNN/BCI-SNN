import random

# BCI
import sys
import os
import warnings
import numpy as np
import numpy.matlib as npm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import KFold

from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.losses import categorical_crossentropy

from bcilib import ssvep_utils as su
warnings.filterwarnings('ignore')

# SNN
from snnlib.spiking_model_pure import *
import time 

# Debug
from pprint import pprint


# %%
def get_training_data(features_data):
    features_data = np.reshape(features_data, (features_data.shape[0], features_data.shape[1], 
                                               features_data.shape[2], 
                                               features_data.shape[3]*features_data.shape[4]))
    train_data = features_data[:, :, 0, :].T
    for target in range(1, features_data.shape[2]):
        train_data = np.vstack([train_data, np.squeeze(features_data[:, :, target, :]).T])

    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 
                                         train_data.shape[2], 1))
    total_epochs_per_class = features_data.shape[3]
    features_data = []
    class_labels = np.arange(CNN_PARAMS['num_classes'])
    labels = (npm.repmat(class_labels, total_epochs_per_class, 1).T).ravel()
    labels = to_categorical(labels)
    
    return train_data, labels


# %%
data_path = os.path.abspath('./data')

CNN_PARAMS = {
    'batch_size': 64,
    'epochs': 50,
    'droprate': 0.25,
    'learning_rate': 0.001,
    'lr_decay': 0.0,
    'l2_lambda': 0.0001,
    'momentum': 0.9,
    'kernel_f': 10,
    'n_ch': 8,
    'num_classes': 12}

FFT_PARAMS = {
    'resolution': 0.2930,
    'start_frequency': 3.0,
    'end_frequency': 35.0,
    'sampling_rate': 256
}

window_len = 1
shift_len = 1
    
all_acc = np.zeros((10, 1))

magnitude_spectrum_features = dict()
complex_spectrum_features = dict()

mcnn_training_data = dict()
ccnn_training_data = dict()

mcnn_results = dict()
ccnn_results = dict()


# %%
all_segmented_data = dict()
for subject in range(0, 10):
    dataset = sio.loadmat(f'{data_path}/s{subject+1}.mat')
    eeg = np.array(dataset['eeg'], dtype='float32')
    
    CNN_PARAMS['num_classes'] = eeg.shape[0]
    CNN_PARAMS['n_ch'] = eeg.shape[1]
    total_trial_len = eeg.shape[2]
    num_trials = eeg.shape[3]
    sample_rate = 256

    filtered_data = su.get_filtered_eeg(eeg, 6, 80, 4, sample_rate)
    #pprint(filtered_data.shape)
    all_segmented_data[f's{subject+1}'] = su.get_segmented_epochs(filtered_data, window_len, 
                                                                  shift_len, sample_rate)
    #pprint(all_segmented_data["s1"].shape)


# %%
for subject in all_segmented_data.keys():
    magnitude_spectrum_features[subject] = su.magnitude_spectrum_features(all_segmented_data[subject], 
                                                                          FFT_PARAMS)
    #pprint(magnitude_spectrum_features[subject].shape)
    complex_spectrum_features[subject] = su.complex_spectrum_features(all_segmented_data[subject], 
                                                                      FFT_PARAMS)
    #pprint(complex_spectrum_features[subject].shape)


# %%
for subject in all_segmented_data.keys():
    mcnn_training_data[subject] = dict()
    ccnn_training_data[subject] = dict()
    train_data, labels = get_training_data(magnitude_spectrum_features[subject])
    mcnn_training_data[subject]['train_data'] = train_data
    mcnn_training_data[subject]['label'] = labels
    
    train_data, labels = get_training_data(complex_spectrum_features[subject])
    ccnn_training_data[subject]['train_data'] = train_data
    ccnn_training_data[subject]['label'] = labels


# %%
for subject in mcnn_training_data.keys():
    dataset = mcnn_training_data[subject]['train_data']
    labels = mcnn_training_data[subject]['label']

    # temp_label = np.zeros((12))
    # for label_index in range(len(labels)):
    #     if not np.array_equal(labels[label_index], temp_label):
    #         temp_label = labels[label_index]
    #         print("Index: ", label_index)
    #         print("Data: ", labels[label_index])
    #         print("\n")

    index_array = list(range(600)) # 600 is the index of first data point that is class 10
    random.shuffle(index_array)

    train_data = np.array([dataset[x] for x in index_array[:420]])
    test_data = np.array([dataset[x] for x in index_array[420:]])
    train_labels = np.array([labels[x, :10] for x in index_array[420:]])
    test_labels = np.array([labels[x, :10] for x in index_array[420:]])

    # print("TRAIN_DATA_MEAD: ", train_data.mean())
    # print("TEST_DATA_MEAD: ", test_data.mean())

    # print("TRAIN DATA: ", train_data)
    # print("TEST_DATA: ", test_data)
    # print("TRAIN_LABELS: ", train_labels)
    # print("TEST_LABELS: ", test_labels)

    names = 'spiking_model'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    acc_record = list([])
    loss_train_record = list([])
    loss_test_record = list([])

    pprint(train_data.shape)
    pprint(labels.shape)

    snn = SCNN()
    snn.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

    for real_epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        for epoch in range(5):
            for i, (images, label) in enumerate(zip(train_data, train_labels)):
                print("NUM: ", i)
                # --- create zoom-in and zoom-out version of each image
                images2 = torch.empty((1, 10, 28, 28))
                img_eeg = torch.from_numpy(np.reshape(images[:, :98, :], (28, 28)))
                for j in range(10):
                    images2[0, j] = img_eeg
                #images2 = torch.empty((images.shape[0] * 2, images.shape[1], images.shape[2]))
                labels2 = torch.empty((1), dtype=torch.int64)
                for j in range(10):
                    if label[j] == 1.:
                        labels2[0] = j
                # ----
                snn.zero_grad()
                optimizer.zero_grad()

                # images2 = images2.div(11.)
                images2[images2[:, :, :] < 1] = 0.
                images2[images2[:, :, :] > 0] = 1.
                print("Mean: ", images2.mean())
                # print("Image2: ", images2)

                images2 = images2.float().to(device)

                # print("MAIN SNN Input shape: ", images2.shape)
                # print(labels2)
                # print(labels2.view(-1, 1))
                # print(torch.zeros(batch_size * 2, 20).scatter_(1, labels2.view(-1, 1), 1))

                # if images2.max() > max_value:
                #     max_value = images2.max()
                # if images2.min() < min_value:
                #     min_value = images2.min()

                # print("max: ", max_value)
                # print("min: ", min_value)

                outputs = snn(images2)
                print(outputs)
                # print(outputs.shape)
                labels_ = torch.zeros(batch_size * 2, 20).scatter_(1, labels2.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print('Real_Epoch [%d/%d], Epoch [%d/%d], Loss: %.5f'
                            %( real_epoch, num_epochs, epoch, 5, running_loss))
                    running_loss = 0
                    print('Time elasped:', time.time() - start_time)

        # ================================== Test ==============================
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
        cm = np.zeros((20, 20), dtype=np.int32)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(zip(test_data, test_labels)):
                print("NUM: ", batch_idx)
                images2 = torch.empty((1, 10, 28, 28))
                img_eeg = torch.from_numpy(np.reshape(images[:, :98, :], (28, 28)))
                for j in range(10):
                    images2[0, j] = img_eeg
                #images2 = torch.empty((images.shape[0] * 2, images.shape[1], images.shape[2]))
                labels2 = torch.empty((1), dtype=torch.int64)
                for j in range(10):
                    if label[j] == 1.:
                        labels2[0] = j

                # images2 = images2.div(11.)
                images2[images2[:, :, :] < 1] = 0.
                images2[images2[:, :, :] > 0] = 1.
                print("Mean: ", images2.mean())
                # print("Image2: ", images2)

                inputs = images2.to(device)
                optimizer.zero_grad()
                outputs = snn(inputs)
                print(outputs)
                labels_ = torch.zeros(batch_size * 2, 20).scatter_(1, labels2.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)
                _, predicted = outputs.cpu().max(1)

                # ----- showing confussion matrix -----

                # cm += confusion_matrix(labels2, predicted)
                # ------ showing some of the predictions -----
                # for image, label in zip(inputs, predicted):
                #     for img0 in image.cpu().numpy():
                #         cv2.imshow('image', img0)
                #         cv2.waitKey(100)
                #     print(label.cpu().numpy())

                total += float(labels2.size(0))
                print("ACCURACY METRICS")
                print(labels2)
                print(labels2.size(0))
                print(total)
                correct += float(predicted.eq(labels2).sum().item())
                print(predicted.eq(labels2).sum().item())
                print(correct)

                if batch_idx % 100 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_data), ' Acc: %.5f' % acc)
        class_names = ['0_ccw', '1_ccw', '2_ccw', '3_ccw', '4_ccw',
                '5_ccw', '6_ccw', '7_ccw', '8_ccw', '9_ccw',
                '0_cw', '1_cw', '2_cw', '3_cw', '4_cw',
                '5_cw', '6_cw', '7_cw', '8_cw', '9_cw']
        # plot_confusion_matrix(cm, class_names)
        print('Iters:', real_epoch, '\n\n\n')
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)
        if real_epoch % 5 == 0:
            print(acc)
            print('Saving..')
            state = {
                'net': snn.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'acc_record': acc_record,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt' + names + '.t7')
            best_acc = acc