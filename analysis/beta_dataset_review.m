load('data/BETA/S1.mat')

size(data.EEG);

load('data/BETA/S2.mat');

size(data.EEG);


for channel = data.EEG(:,1,1,1)
    channel;
end