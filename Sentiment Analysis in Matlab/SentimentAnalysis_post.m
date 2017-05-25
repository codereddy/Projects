%This script takes tweets and return positive, negative or neutral
%sentiment for each tweet.
%It uses Naive Bayes classifier to classify the tweets.

filename='Tweets.csv';
Data=readtable(filename);
Data=table2cell(Data);
 Tweets=Data(:,1);
 
 if size(Tweets,1)==0
     fprintf('There are no Tweets to Analyze')
     return
 end
 
sentiment_class=cell(size(Tweets,1),2);
confidence = zeros(size(Tweets,1),1);

 
%splitting each tweet into words
for i =1:size(Tweets,1)
    Tweets{i,:}=strsplit(char(Tweets{i,:}));
end

%unpack the Already Trained Bayes Model variable
%load('BayesModel.mat')

%predict the sentiment for the tweets using the model
for i=1:size(Tweets,1)
    sentiment_class(i,1) = Data(i,1);
    [sentiment_class(i,2),post_temp,~]= predict(Mdl,double(ismember(Mdl.PredictorNames,Tweets{i,:})));
if(post_temp(1,1) > post_temp(1,2) && post_temp(1,1)>=0.7)
    sentiment_class{i,2} = 'Negative';
    confidence(i,1) = post_temp(1,1);
elseif (post_temp(1,1) < post_temp(1,2) && post_temp(1,2)>=0.7)
    sentiment_class{i,2}  = 'Positive';
    confidence(i,1) = post_temp(1,2);
else
    if(post_temp(1,1) > post_temp(1,2))
        confidence(i,1) = post_temp(1,1);
    else
        confidence(i,1) = post_temp(1,2);
    end
    sentiment_class{i,2} = 'Neutral';
end

% fprintf('Tweet: "%s"\nSentiment:    %s\nConfidence:    %.2f\n',...
%     strjoin(Tweets{i,:},' '),label{1},post(strcmp(Mdl.ClassNames,label)))
end
