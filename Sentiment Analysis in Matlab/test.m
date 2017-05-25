%  filename='TwitterTrainingData.csv';
% % Data=readtable(filename);
% % Data=table2cell(Data);
examples=Data(:,4);
test_examples = examples(10001:20000,1);

% splitting sentences into words
for i =1:size(test_examples,1)
    test_examples{i,:}=strsplit(char(test_examples{i,:}));
end

sentiment_class=cell(size(test_examples,1),2);

sentiment_class(:,1) = Data(10001:20000,2);

post = zeros(size(test_examples,1),2);


% tokens = unique([test_examples{:}]);
% X = zeros(length(test_examples),length(tokens));
% for i = 1:length(test_examples)
%     X(i,:) = ismember(tokens,test_examples{i});
% end
% 
% Mdl = fitcnb(X,sentiment_class,'Distribution','mn','PredictorNames',tokens);

 %heck the trained model with the test example.
for i=1: size(test_examples,1)
[sentiment_class(i,2),posti,~]= predict(Mdl,double(ismember(Mdl.PredictorNames,test_examples{i})));
post(i,:) = posti;
% fprintf('Test example: "%s"\nSentiment:    %s\nPosterior:    %.2f\n',...
%     strjoin(test_examples{i},' '),sentiment_class{i,2},posti(strcmp(Mdl.ClassNames,label)))
end