
%% Extract text data from files
% filename = "DataSetAmazon.csv";
% data = readtable(filename,'TextType','string');
%To index : opinions -> data.Var1  sentiments-> data.Var2
%USING BIGGER DATASET (Full-Amazon)
% data = importdata("DATASET.mat");
 %Cut the table to reduce time of training
% nsamples = 3001;
 %data([nsamples:401000],:)=[]; %We keep nsamples
 %Import Amazon(short)+IMDB+YELP
data = importdata("VALIDATIONTABLE_AMAZON_IMDB_YELP.mat");
data1 = importdata("TRAININGTABLE_AMAZON_IMDB_YELP.mat");
%I had them separete but as I have separation module after that I join now
%the two datasets
data = [data;data1];
%% Remove rows without text
idxEmpty = strlength(data.Var1) == 0;
data(idxEmpty,:) = [];
%% Prepare data sets
data.Var2 = categorical(data.Var2);
%Dividing data in 70%training,15%validation and 15%test
cvp = cvpartition(data.Var2,'Holdout',0.3);
dataTrain = data(training(cvp),:);
dataHeldOut = data(test(cvp),:);
cvp = cvpartition(dataHeldOut.Var2,'Holdout',0.5);
dataValidation = dataHeldOut(training(cvp),:);
dataTest = dataHeldOut(test(cvp),:);
%% Extract the data from partitions
textDataTrain = dataTrain.Var1;
textDataValidation = dataValidation.Var1;
textDataTest = dataTest.Var1;
YTrain = dataTrain.Var2;
YValidation = dataValidation.Var2;
YTest = dataTest.Var2;
%% Prepocess the Text Data
documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation);
%For more treatment look at RNNproject.m
%% Convert sentences into sequences of numeric indices (word encoding)
%in order to introduce in LSTM network
enc = wordEncoding(documentsTrain);
%Explore documents lengths
documentLengths = doclength(documentsTrain);
%To determine where truncate (there are some wordsencoded few times)
% figure
% histogram(documentLengths)
% title("Document Lengths")
% xlabel("Length")
% ylabel("Number of Documents")
%Set the appropiate length to truncate
n_length = 15; %I selected 100 (appears more than 3000)

XTrain = doc2sequence(enc,documentsTrain,'Length',n_length);
%The value n_length is in which you truncate (see documentation) I take all
%without truncating sentence
XValidation = doc2sequence(enc,documentsValidation,'Length',n_length);
% %% Create and train LSTM network
inputSize = 1;
embeddingDimension = 100;%100
numWords = enc.NumWords;
numHiddenUnits = 50;%180-->Reduce to reduce overfitting
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer()
    fullyConnectedLayer(numClasses)
    dropoutLayer()
    fullyConnectedLayer(numClasses)
    dropoutLayer()
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
options = trainingOptions('adam', ...
    'MaxEpochs',7, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);
net = trainNetwork(XTrain,YTrain,layers,options);
%% Test LSTM Network
%Preprocess data
documentsTest = preprocessText(textDataTest);
XTest = doc2sequence(enc,documentsTest,'Length',n_length);
%Clasifying using the trained LSTM network
YPred = classify(net,XTest);
accuracy = sum(YPred == YTest)/numel(YPred)
%% PreprocessTextfunction
function documents = preprocessText(textData)

% Tokenize the text.
documents = tokenizedDocument(textData);

% Convert to lowercase.
documents = lower(documents);

% Erase punctuation.
documents = erasePunctuation(documents);

% Remove stop words
% documents = removeStopWords(documents);

end


