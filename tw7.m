train = readmatrix('mnist_test.csv.xlsx');
test = readmatrix('mnist_test.csv.xlsx');
XTrain = train(:, 2:end);
YTrain = categorical(train(:, 1));
XTest = test(:, 2:end);
YTest = categorical(test(:, 1));
XTrain = reshape(XTrain', 28, 28, 1, []);
XTrain = permute(XTrain, [2 1 3 4]);
XTest = reshape(XTest', 28, 28, 1, []);
XTest = permute(XTest, [2 1 3 4]);
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 64, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);

YPred = classify(net, XTest);
accuracy = mean(YPred == YTest) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

confusionchart(YTest, YPred);
