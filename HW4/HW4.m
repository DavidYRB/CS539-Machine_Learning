%% Bayesian Networks
clc; clear; close all;

% Naive Bayes Model
% Data input and stratify
raw_data = readtable('adult.data.csv');
temp_idx = [];
for i = 1:height(raw_data)
    idx = strcmp(table2array(raw_data(i,15)),'>50K');
    temp_idx = [temp_idx;idx];
end

rng(1)
% Find indices of instances of different classes
class_1 = find(temp_idx==1);
class_2 = find(temp_idx ==0);
% Divide instances of a single class into training and testing part
train_1 = datasample(class_1,round(0.75*length(class_1)),'Replace',false);
test_1 = setdiff(class_1,train_1);
train_2 = datasample(class_2,round(0.75*length(class_2)),'Replace',false);
test_2 = setdiff(class_2,train_2);
% merge into a single train and test dataset
Train_set = [raw_data(train_1,:);raw_data(train_2,:)];
Test_set = [raw_data(test_1,:);raw_data(test_2,:)];

% train naive bayes model
Md1 = fitcnb(Train_set(:,1:14),Train_set(:,15));
% Predict 
label = predict(Md1,Test_set(:,1:14));
count = 0
te = table2array(Test_set(:,15));

for i = 1:length(label)
   if strcmp(label{i},te{i})
       count = count+1;
   end
end
confumat = confusionmat(te,label)
accu = count/length(label)

%% Bayesion Network
