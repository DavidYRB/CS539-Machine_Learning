clc; clear; close all;
%% Data load
% Load data
temp = readtable('adult.data.csv');

age = table2array(temp(:,1));
workclass = table2array(temp(:,2));
fnlwgt = table2array(temp(:,3));
education = table2array(temp(:,4));
education_num = table2array(temp(:,5));
marital_status = table2array(temp(:,6));
occupation = table2array(temp(:,7));
relationship = table2array(temp(:,8));
race = table2array(temp(:,9));
sex = table2array(temp(:,10));
capital_gain = table2array(temp(:,11));
capital_loss = table2array(temp(:,12));
hours_per_week = table2array(temp(:,13));
native_country = table2array(temp(:,14));
class = table2array(temp(:,15));
% original dataset with labels
Data_raw = table(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,...
    race,sex,capital_gain,capital_loss,hours_per_week,native_country,class);
% Train = Data_raw(1:0.8*height(Data_raw),:);
% Test = Data_raw(1:0.2*height(Data_raw),:);
Data = Data_raw(:,1:14);
Class = Data_raw(:,15);
% Using five group parameters to generate classification tree with PRUNE OFF
para_group = [6,50,2; 6,20,2; 8,50,2; 8,20,2; 6,50,6]; % parameter for 'MaxNumSplits', 'MinParentSize','MinLeafSize'
Class = table2cell(Class);

SVMModel = fitcsvm(Data,Class);