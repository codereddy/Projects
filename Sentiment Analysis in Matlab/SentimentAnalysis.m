filename='TwitterTrainingData.csv';
Data=readtable(filename);
Data=table2cell(Data);
 training_exmples=Data(:,4);
training_exmples_t = training_exmples(1:10000,1);
%splitting sentences into words
for i =1:size(training_exmples_t,1)
    training_exmples_t{i,:}=strsplit(char(training_exmples_t{i,:}));
end

%class of labels stored in variable
sentiment_class=Data(1:size(training_exmples_t,1),2);

%Turn the training data into a numeric matrix. This is a bag-of-words model that uses frequency of words as predictors.
tokens = unique([training_exmples_t{:}]);
X = zeros(length(training_exmples_t),length(tokens));
for i = 1:length(training_exmples_t)
    X(i,:) = ismember(tokens,training_exmples_t{i});
end

%fit the model using bayes classifier
Mdl = fitcnb(X,sentiment_class,'Distribution','mn','PredictorNames',tokens);
%save the trained model for later
save 'BayesModel.mat' Mdl -v7.3;

%check the trained model with the test example.
test_example = {'I','am', 'not', 'happy'};
[label,post,~]= predict(Mdl,double(ismember(Mdl.PredictorNames,test_example)));
fprintf('Test example: "%s"\nSentiment:    %s\nPosterior:    %.2f\n',...
    strjoin(test_example,' '),label{1},post(strcmp(Mdl.ClassNames,label)))