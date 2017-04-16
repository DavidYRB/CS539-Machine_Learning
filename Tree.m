clc; clear; close all;
%% Section A: Trees
% Classification Trees
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

for i = 1:5
    disp(i)
    tic
    tree1 = fitctree(Data,Class,'MaxNumSplits',para_group(i,1),'CrossVal','on','Kfold',4,...
        'MinParentSize',para_group(i,2),'MinLeafSize',para_group(i,3),'Prune', 'off');
    view(tree1.Trained{1},'Mode','graph')
    toc
    pre = kfoldPredict(tree1);
    temp = 0;
    % Calculate accuracy
    for j = 1:length(pre)
        if strcmp(Class{j},pre{j}) == 1
            temp = temp +1;
        end
    end
    accu = temp/length(pre)
    % Calculate size of tree
    numBranches = @(x)sum(x.IsBranch);
    treeNumSplits = cellfun(numBranches, tree1.Trained)
end


for i = 1:5
    disp(i)
    tic
    tree1_p = fitctree(Data,Class,'MaxNumSplits',para_group(i,1),'CrossVal','on','Kfold',4,...
        'MinParentSize',para_group(i,2),'MinLeafSize',para_group(i,3),'Prune', 'on');
    view(tree1_p.Trained{1},'Mode','graph')
    toc
    pre_p = kfoldPredict(tree1_p);
    temp = 0;
    % Calculate accuracy
    for j = 1:length(pre_p)
        if strcmp(Class{j},pre_p{j}) == 1
            temp = temp +1;
        end
    end
    accu = temp/length(pre)
    % Calculate size of tree
    numBranches_p = @(x)sum(x.IsBranch);
    treeNumSplits_p = cellfun(numBranches_p, tree1_p.Trained)
end
%% Random forest
rand = fitensemble(Data,Class,'Bag','AllPredictorCombinations','Tree','CrossVal','on','Type','Classification')

%% Regression Tree
Data1 = table(age,workclass,fnlwgt,education,marital_status,occupation,relationship,...
    race,sex,capital_gain,capital_loss,hours_per_week,native_country,class);
Class1 = table(education_num);
para_group = [6,50,2; 6,20,2; 8,50,2; 8,20,2; 6,50,6]; % parameter for 'MaxNumSplits', 'MinParentSize','MinLeafSize'

for i = 1:5
    disp(i)
    tic
    tree2 = fitrtree(Data1,Class1,'MaxNumSplits',para_group(i,1),'Kfold',4,'MinParentSize',para_group(i,2),...
        'MinLeafSize',para_group(i,3),'Prune', 'off')
    view(tree2.Trained{1},'Mode','graph')
    toc
    [SSE,RMSE,RSE,R_sq] = evaluation(tree2.Y, kfoldPredict(tree2))

end

for i = 1:5
    disp(i)
    tic
    tree2_p = fitrtree(Data1,Class1,'MaxNumSplits',para_group(i,1),'Kfold',4,'MinParentSize',para_group(i,2),...
        'MinLeafSize',para_group(i,3),'Prune', 'off')
    view(tree2_p.Trained{1},'Mode','graph')
    toc
    [SSE,RMSE,RSE,R_sq] = evaluation(tree2_p.Y, kfoldPredict(tree2_p))

end

