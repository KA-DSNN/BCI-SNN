% Student name = Nogay K?PEL?O?LU
% Student number = 03714831
clc
close all;
clear all;

data_path = 'C:\\Users\\kupel\\OneDrive\\Desktop\\NISE\\NISE_WS1920_BCI_tutorial\\data\\calibration_unprocessed.set';
data_processed = 'C:\\Users\\kupel\\OneDrive\\Desktop\\NISE\\NISE_WS1920_BCI_tutorial\\data\\calibration.set';
calibration_unprocessed = pop_loadset(data_path);
calibration_processed = pop_loadset(data_processed);
eegplot(calibration_processed.data, 'data2', calibration_unprocessed.data,'eloc_file', calibration_processed.chanlocs);