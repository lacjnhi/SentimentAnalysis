# This program predicts if the stock price of a company will increase or decrease
# Based on top news headlines

# Import libraries
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Store the data into variables
df1 = pd.read_csv('Data/Dow_Jones_Industrial_Average_News.csv')
df2 = pd.read_csv('Data/Dow_Jones_Industrial_Average_Stock.csv')

# Show the first 3 rows of data for df1
# print(df1.head(3))

# Get the number of rows and columns for df1
# print(df1.shape)

# Print the first 3 rows of data for df2
# print(df2.head(3))

# Get the number of rows and columns for df2
# print(df2.shape)

# Merge the data set on the data field
merge = df1.merge(df2, how='inner', on='Date', left_index= True)

# Show the merge data
# print(merge)

# Combine the top news headlines
headlines = []

for row in range(0, len(merge.index)):
    headlines.append(' '.join(str(x) for x in merge.iloc[row, 2:27]))

# Print a sample of the combined headlines
# print(headlines[0])

# Clean the data
clean_headlines = []

for i in range(0, len(headlines)):
    clean_headlines.append(re.sub("b[(')]", '', headlines[i])) # remove b'
    clean_headlines[i] = re.sub('b[(")]', '', clean_headlines[i]) # remove b"
    clean_headlines[i] = re.sub("\'", '', clean_headlines[i]) # remove \'

# Show the combined cleaned headlines
# print(clean_headlines[20])

# Add the cleaned headlines to the merge data set
merge['Combined_News'] = clean_headlines

# Show the new column
# print(merge['Combined_News'][0])

# SHow the first 3 rows of the merge data set
# print(merge.head(3))

# Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create two new columns 'Subjectivity' and 'Polarity'
merge['Subjectivity'] = merge['Combined_News'].apply(getSubjectivity)
merge['Polarity'] = merge['Combined_News'].apply(getPolarity)

# Show the new columns in the merge data set
''' 2 more columns 'Subjectivity' and 'Polarity' are shown
Subjectivity: 0 objective; 1 subjective (scale 0 to 1)
Polarity    : -1 negative; 1 positive  (scale -1 to 1)
'''
# print(merge.head(3))

# Create a function to get the sentiment scores
def getSIA(text): # SIA: Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Get the sentiment scores for each day
compound = []
neg = [] # negative
pos = [] # positive
neu = [] # neutral
SIA = 0

for i in range(0, len(merge['Combined_News'])):
    SIA = getSIA(merge['Combined_News'][i]) # sentiment score
    # compound: a score that calculates the sum of all the lexicon ratings
    # which have been normalized between -1: most extreme negative and 1: most extreme positive
    compound.append(SIA['compound'])
    neg.append(SIA['neg'])
    neu.append(SIA['neu'])
    pos.append(SIA['pos'])

# Store the sentiment scores in the merge data set
merge['Compound'] = compound
merge['Negative'] = neg
merge['Neutral']  = neu
merge['Positive'] = pos

# Show the merge data
# print(merge.head(3))

# Create a list of columns to keep
keep_columns = ['Open', 'High', 'Low', 'Volume', 'Subjectivity', 'Polarity', 'Compound', 'Negative', 'Neutral', 'Positive', 'Label']
df = merge[keep_columns]
# print(df)

# Create the feature data set
x = df
x = np.array(x.drop(['Label'], 1))

# Create the target data set
y = np.array(df['Label'])

# Split the data into 80% training and 20% testing data sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

# Create and train the model
model = LinearDiscriminantAnalysis().fit(x_train, y_train)

# Show the model predictions
predictions = model.predict(x_test)
print(predictions)  # both models don't match that accurately
print (y_test)      # around 84% accuracy

# Show the model metrics
print(classification_report(y_test, predictions)) #
# 84% accurate
