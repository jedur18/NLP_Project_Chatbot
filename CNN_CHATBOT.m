
%% Load pretrained word embedding
emb = fastTextWordEmbedding; % Returns 300-dimensional pretrained word embedding for 1 million english words
%This process is slowm so when you have it in workspace comment it if you
%want to make more tryals.
%% Load Data
%filenameTrain =  "CNNchatbotAMAZONTrainingdataset.csv"; %Amazon dataset
%filenameTrain =  "CNNchatbotTWITTERTrainingdataset.csv"; %Twitter dataset
filenameTrain =  "CNNchatbotAMAZON+IMDBTrainingdataset.csv"; % Amazon (Big+small) %The name of csv is wrong
%filenameTrain =  "CNNchatbotAMAZON_IMDB_YELPTrainingdataset.csv";
textName = "Var1";
labelName = "Var2";
ttdsTrain = tabularTextDatastore(filenameTrain,'SelectedVariableNames',[textName labelName]);
% ttdsTrain.ReadSize
% preview(ttdsTrain)
% Creating table with the data
labels = readLabels(ttdsTrain,labelName);
classNames = unique(labels);
numObservations = numel(labels);
%% Transform the data with sequence of length sequenceLength
sequenceLength = 100;
tdsTrain = transform(ttdsTrain,@(data) transformTextData(data,sequenceLength,emb,classNames))
%% Creating transformed datastore with validation data
%filenameValidation =  "CNNchatbotAMAZONValidationdataset.csv";
%filenameValidation =  "CNNchatbotTWITTERValidationdataset.csv";
filenameValidation = "CNNchatbotAMAZON+IMDBValidationdataset.csv"; %it is full amazon, name is wrong
%filenameValidation =  "CNNchatbotAMAZON_IMDB_YELPValidationdataset.csv"; 
ttdsValidation = tabularTextDatastore(filenameValidation,'SelectedVariableNames',[textName labelName]);
tdsValidation = transform(ttdsValidation, @(data) transformTextData(data,sequenceLength,emb,classNames))
%% Network architecture
numFeatures = emb.Dimension;
inputSize = [1 sequenceLength numFeatures];
numFilters = 200;
ngramLengths = [2 3 4 5];
numBlocks = numel(ngramLengths);
numClasses = numel(classNames);
%Creating layer graph containing the input layer
layer = imageInputLayer(inputSize,'Normalization','none','Name','input');
lgraph = layerGraph(layer);
% For each of the n-grams, create a block convolution, batch normalization,
% ReLU, dropout and max pooling layers. All blocks are connected to the
% input layer
for k = 1:numBlocks
    N = ngramLengths(k);
    block = [
        convolution2dLayer([1 N],numFilters,'Name',"conv"+N,'Padding','same')
        batchNormalizationLayer('Name',"batch_norm"+N)
        reluLayer('Name',"relu"+N)
        dropoutLayer(0.2,'Name',"drop"+N) 
        maxPooling2dLayer([1 sequenceLength],'Name',"max_pool"+N)];
    lgraph = addLayers(lgraph,block);
    lgraph = connectLayers(lgraph,'input',"conv"+N);
end
%Architecture network (layout)
% figure
% plot(lgraph)
% title("Architecture of the Network")

%Adding depth concatenation layer, the fully connected layer, the softmax
%layer and the classification layer
layers = [
    depthConcatenationLayer(numBlocks,'Name','depth_concat')
    fullyConnectedLayer(numClasses,'Name','fully_conect')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
%Join with the previous layers (no conects it)
lgraph = addLayers(lgraph,layers);
%Visualization of new arquitecture
% figure
% plot(lgraph)
% title("Architecture of the Network")

%Conection of the max pooling layers to the depth concatenation layer
for k = 1:numBlocks
    N = ngramLengths(k);
    lgraph = connectLayers(lgraph,"max_pool"+N,"depth_concat/in"+k);
end
% VIsualization of full network connected
figure
plot(lgraph)
title("Architecture of the Network")
%% Training the Network
%Train n_epoch epochs with mini-batch size of miniBatchSize
n_epoch = 4;
miniBatchSize = 1000 ;  %20 for 1k amazon dataset
numIterationsPerEpoch = floor(numObservations/miniBatchSize);
options = trainingOptions('adam',... %Options available for adam optimizer: https://se.mathworks.com/help/deeplearning/ref/nnet.cnn.trainingoptionsadam.html
    'MaxEpochs',n_epoch, ...
    'InitialLearnRate',0.001, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','never', ... %The dataset is already suffle
    'ValidationData',tdsValidation, ...
    'ValidationFrequency',numIterationsPerEpoch, ... %Validates the network at each epoch
    'Plots','training-progress', ... %Display training progress
    'Verbose',false); %Not show verbose
net = trainNetwork(tdsTrain,lgraph,options);
%  %% TESTING NETWORK (COMMENT THE NETWORK TRAINING,previous section)
%  filenameTest = "CNNchatbotAMAZON+IMDBTestingdataset.csv";
% % filenameTest = "CHATBOT_dialog_sentences.csv";
% ttdsTest = tabularTextDatastore(filenameTest,'SelectedVariableNames',[textName labelName]);
% tdsTest = transform(ttdsTest, @(data) transformTextData(data,sequenceLength,emb,classNames))
% labelsTest = readLabels(ttdsTest,labelName);
% YTest = categorical(labelsTest,classNames);
% %Importing trained network (the name of variable is net)
% load('C:\Users\eri53\Desktop\MASTER\SEGUNDO\Asignaturas\Neural Networks and Learning Machines\Project\Matlab\BETTER NETWORKS ACHIEVED\CNN(4poch_92.41ac)\NETWORK\NETWORK.mat')
% % Testing
% YPred = classify(net,tdsTest);
% accuracy = mean (YPred == YTest)
% %Print the confusion matrix (confusionmat(KnownGroup,PredictionGroup)
% ConfMatrix = confusionmat(YTest,YPred)
% Accuracy = (ConfMatrix(1,1)+ConfMatrix(2,2))/sum(sum(ConfMatrix))%(TP+TN)/(TP+FP+FN+TN)
% Precision = ConfMatrix(1,1)/sum(ConfMatrix(:,1)) %TP/TP+FP
% Recall = ConfMatrix(1,1)/sum(ConfMatrix(1,:)) %TP/TP+FN
% F1Score = 2*(Recall*Precision)/(Recall+Precision) %weighted average of precision and recall (to have FP and FN into account)
% Metrics = table(Accuracy,Precision,Recall,F1Score)
%% Functions
function labels = readLabels(ttds,labelName)

ttdsNew = copy(ttds);
ttdsNew.SelectedVariableNames = labelName;
tbl = readall(ttdsNew);
labels = tbl.(labelName);

end

function dataTransformed = transformTextData(data,sequenceLength,emb,classNames)

% Preprocess documents.
textData = data{:,1};
textData = lower(textData);
documents = tokenizedDocument(textData);

% Convert documents to embeddingDimension-by-sequenceLength-by-1 images.
predictors = doc2sequence(emb,documents,'Length',sequenceLength);

% Reshape images to be of size 1-by-sequenceLength-embeddingDimension.
predictors = cellfun(@(X) permute(X,[3 2 1]),predictors,'UniformOutput',false);

% Read labels.
labels = data{:,2};
responses = categorical(labels,classNames);

% Convert data to table.
dataTransformed = table(predictors,responses);

end
