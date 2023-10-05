# %%
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image 
import random
import matplotlib
matplotlib.style.use('ggplot')

from sklearn.metrics import confusion_matrix
import seaborn as sns

import re


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
# from nltk.tokenize import word_tokenize



# %%
# !pip install ntlk

# %%
def normal_word_tokenize(text):
    return text.split()

# %%
def preprocess_tweet(text):
    # words = normal_word_tokenize(text)
    # words = text.split()
    # filtered_words = [word for word in words if "t.co" not in word]
    # filtered_words = [word for word in words if "t.co" not in word]
    # final_text =  ' '.join(filtered_words)
    # final_text.replace('Â', "'")
    # final_text = re.sub( r'&\w+;', '', final_text)
    return text
    

# %%
train_df = pd.read_csv('Corona_train.csv')
train_df['CoronaTweet'] = train_df['CoronaTweet'].apply(preprocess_tweet)
train_df.head()

# %%
all_words = set()

for sentence in train_df['CoronaTweet']:
    for word in normal_word_tokenize(sentence):
        all_words.add(word)
        
all_words_len = len(all_words)

# %%
train_df_positive = train_df[train_df['Sentiment'] == 'Positive']
positive_wc = {}

sentences = train_df_positive['CoronaTweet']
total_positive_words = 0


for sentence in sentences:
    words = normal_word_tokenize(sentence)
    total_positive_words += len(words)
    
    for word in words:
        if word not in positive_wc:
            positive_wc[word] = 0
        positive_wc[word] += 1  

# %%
train_df_negative = train_df[train_df['Sentiment'] == 'Negative']
negative_wc = {}

sentences = train_df_negative['CoronaTweet']
total_negative_words = 0

for sentence in sentences:
    words = normal_word_tokenize(sentence)
    total_negative_words += len(words)
    for word in words:
        if word not in negative_wc:
            negative_wc[word] = 0
        negative_wc[word] += 1  

# %%
train_df_neutral = train_df[train_df['Sentiment'] == 'Neutral']
neutral_wc = {}

sentences = train_df_neutral['CoronaTweet']

total_neutral_words = 0

for sentence in sentences:
    words = normal_word_tokenize(sentence)
    total_neutral_words += len(words)    
    for word in words:
        if word not in neutral_wc:
            neutral_wc[word] = 0
        neutral_wc[word] += 1  

# %%
positive_prob = {}

for word, count in positive_wc.items():
    positive_prob[word] = np.log((count + 1)) - np.log(total_positive_words + all_words_len)
    
negative_prob = {}

for word, count in negative_wc.items():
    negative_prob[word] = np.log((count + 1)) - np.log(total_negative_words + all_words_len)
    
neutral_prob = {}

for word, count in neutral_wc.items():
    neutral_prob[word] = np.log((count + 1)) - np.log(total_neutral_words + all_words_len)
    
# Prior Probabilities, for them laplace smoothing is not that useful
y_is_positive = np.log(len(train_df_positive)) - np.log(len(train_df))
y_is_negative = np.log(len(train_df_negative)) - np.log(len(train_df))
y_is_neutral = np.log(len(train_df_neutral)) - np.log(len(train_df))

# %%
predictions = []
actual = []
for id, row in train_df.iterrows():
    
    words = normal_word_tokenize(row['CoronaTweet'])
    
    positive_sum = y_is_positive
    negative_sum = y_is_negative
    neutral_sum = y_is_neutral
    
    for word in words:
        # 0.5 due to laplace smoothing
        if word not in all_words:
            continue
        
        positive_sum += positive_prob.get(word, -np.log(all_words_len + total_positive_words))
        negative_sum += negative_prob.get(word,  -np.log(all_words_len + total_negative_words))
        neutral_sum += neutral_prob.get(word,  -np.log(all_words_len + total_neutral_words))
    
    if positive_sum >= negative_sum and positive_sum >= neutral_sum:
        predictions.append('Positive')
    elif neutral_sum >= positive_sum and neutral_sum >= negative_sum:
        predictions.append('Neutral')   
    elif negative_sum >= positive_sum and negative_sum >= neutral_sum:
        predictions.append('Negative')
     
    
    actual.append(row['Sentiment'])
    
    print(actual[-1], predictions[-1], positive_sum, negative_sum, neutral_sum, row['CoronaTweet'])


# %%
corrrect = 0
for ac, pred in zip(actual, predictions):
    if ac == pred:
        corrrect += 1
        
print(corrrect / len(actual))

# %%
actual_train = actual.copy()
predictions_train = predictions.copy()

# %%
validate_df = pd.read_csv('Corona_validation.csv')
validate_df['CoronaTweet'] = validate_df['CoronaTweet'].apply(preprocess_tweet)
validate_df.head()

# %%
predictions = []
actual = []
for id, row in validate_df.iterrows():
    
    words = normal_word_tokenize(row['CoronaTweet'])
    
    positive_sum = y_is_positive
    negative_sum = y_is_negative
    neutral_sum = y_is_neutral
    
    for word in words:
        # 0.5 due to laplace smoothing
        if word not in all_words:
            continue
        
        positive_sum += positive_prob.get(word, -np.log(all_words_len + total_positive_words))
        negative_sum += negative_prob.get(word,  -np.log(all_words_len + total_negative_words))
        neutral_sum += neutral_prob.get(word,  -np.log(all_words_len + total_neutral_words))
    
    if positive_sum >= negative_sum and positive_sum >= neutral_sum:
        predictions.append('Positive')
    elif neutral_sum >= positive_sum and neutral_sum >= negative_sum:
        predictions.append('Neutral')   
    elif negative_sum >= positive_sum and negative_sum >= neutral_sum:
        predictions.append('Negative')
     
    
    actual.append(row['Sentiment'])
    
    print(actual[-1], predictions[-1], positive_sum, negative_sum, neutral_sum, row['CoronaTweet'])


# %%
corrrect = 0
for ac, pred in zip(actual, predictions):
    if ac == pred:
        corrrect += 1
        
print(corrrect / len(actual))

# %%
stopwords = set(STOPWORDS)

neutral_cloud = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

entire_text = ' '.join(train_df_neutral['CoronaTweet'].tolist())

neutral_cloud.generate(entire_text)

plt.imshow(neutral_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
stopwords = set(STOPWORDS)

positive_cloud = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

entire_text = ' '.join(train_df_positive['CoronaTweet'].tolist())

positive_cloud.generate(entire_text)

plt.imshow(positive_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
stopwords = set(STOPWORDS)

negative_cloud = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

entire_text = ' '.join(train_df_negative['CoronaTweet'].tolist())

negative_cloud.generate(entire_text)

plt.imshow(negative_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# Part b:

# %%
options = ["Neutral", "Positive","Negative"]
random_prediction = [random.choice(options) for _ in range(len(actual))]
corrrect = 0
for ac, pred in zip(actual, random_prediction):
    if ac == pred:
        corrrect += 1
        
print(corrrect / len(actual))

# %%
all_positive = ["Positive" for _ in range(len(actual))]
corrrect = 0
for ac, pred in zip(actual, all_positive):
    if ac == pred:
        corrrect += 1
        
print(corrrect / len(actual))

# %% [markdown]
# Part c
# 

# %%
confusion_mat = confusion_matrix(actual_train, predictions_train, labels=['Positive', 'Negative', 'Neutral'])

# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d',cmap='Blues', xticklabels=['Positive', 'Negative', 'Neutral'], yticklabels=['Positive', 'Negative', 'Neutral'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix on Training Data')
plt.show()

# %%
confusion_mat = confusion_matrix(actual, predictions, labels=['Positive', 'Negative', 'Neutral'])

# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative', 'Neutral'], yticklabels=['Positive', 'Negative', 'Neutral'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix on Validation Data')
plt.show()

# %%
confusion_mat = confusion_matrix(actual, all_positive, labels=['Positive', 'Negative', 'Neutral'])

# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative', 'Neutral'], yticklabels=['Positive', 'Negative', 'Neutral'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix on Positive prediction on Validation Data')
plt.show()

# %%
confusion_mat = confusion_matrix(actual, random_prediction, labels=['Positive', 'Negative', 'Neutral'])

# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative', 'Neutral'], yticklabels=['Positive', 'Negative', 'Neutral'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix on Random Prediction on Validation Data')
plt.show()

# %%
all_words = set()

for sentence in train_df['CoronaTweet']:
    words = normal_word_tokenize(sentence)
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    for word in words:
        all_words.add(word)
        
all_words_len = len(all_words)

# %%
train_df_positive = train_df[train_df['Sentiment'] == 'Positive']
positive_wc = {}

total_positive_words = 0

sentences = train_df_positive['CoronaTweet']

for sentence in sentences:
    words = normal_word_tokenize(sentence)
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    total_positive_words += len(words)
    
    for word in words:
        if word not in positive_wc:
            positive_wc[word] = 0
        positive_wc[word] += 1  

# %%
train_df_negative = train_df[train_df['Sentiment'] == 'Negative']
negative_wc = {}

sentences = train_df_negative['CoronaTweet']
total_negative_words = 0

for sentence in sentences:
    words = normal_word_tokenize(sentence)
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    total_negative_words += len(words)
    
    
    for word in words:
        if word not in negative_wc:
            negative_wc[word] = 0
        negative_wc[word] += 1  

# %%
train_df_neutral = train_df[train_df['Sentiment'] == 'Neutral']
neutral_wc = {}

sentences = train_df_neutral['CoronaTweet']
total_neutral_words = 0


for sentence in sentences:
    words = normal_word_tokenize(sentence)
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    total_neutral_words += len(words)
    
    
    for word in words:
        if word not in neutral_wc:
            neutral_wc[word] = 0
        neutral_wc[word] += 1  

# %%
positive_prob = {}

for word, count in positive_wc.items():
    positive_prob[word] = np.log((count + 1)) - np.log(total_positive_words + all_words_len)
    
negative_prob = {}

for word, count in negative_wc.items():
    negative_prob[word] = np.log((count + 1)) - np.log(total_negative_words + all_words_len)
    
neutral_prob = {}

for word, count in neutral_wc.items():
    neutral_prob[word] = np.log((count + 1)) - np.log(total_neutral_words + all_words_len)  
    
# Prior Probabilities, for them laplace smoothing is not that useful
y_is_positive = np.log(len(train_df_positive) / len(train_df))
y_is_negative = np.log(len(train_df_negative) / len(train_df))
y_is_neutral = np.log(len(train_df_neutral) / len(train_df))

# %%
predictions = []
actual = []
for id, row in validate_df.iterrows():
    
    words = normal_word_tokenize(row['CoronaTweet'])
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    positive_sum = y_is_positive
    negative_sum = y_is_negative
    neutral_sum = y_is_neutral
    
    for word in words:
        positive_sum += positive_prob.get(word,-np.log(all_words_len + total_positive_words))
        negative_sum += negative_prob.get(word,  -np.log(all_words_len + total_negative_words))
        neutral_sum += neutral_prob.get(word,  -np.log(all_words_len + total_neutral_words))
    
    if positive_sum >= negative_sum and positive_sum >= neutral_sum:
        predictions.append('Positive')
    elif negative_sum >= positive_sum and negative_sum >= neutral_sum:
        predictions.append('Negative')
    elif neutral_sum >= positive_sum and neutral_sum >= negative_sum:
        predictions.append('Neutral')    
    
    actual.append(row['Sentiment'])


# %%
corrrect = 0
for ac, pred in zip(actual, predictions):
    if ac == pred:
        corrrect += 1
        
print(corrrect / len(actual))

# %%
stopwords = set(STOPWORDS)

positive_cloud = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

entire_text = ' '.join(train_df_positive['CoronaTweet'].tolist())
words = normal_word_tokenize(entire_text)
stemmed_words = [stemmer.stem(word) for word in words]
entire_text = ' '.join(stemmed_words)

positive_cloud.generate(entire_text)

plt.imshow(positive_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
stopwords = set(STOPWORDS)

negative_cloud = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

entire_text = ' '.join(train_df_negative['CoronaTweet'].tolist())
words = normal_word_tokenize(entire_text)
stemmed_words = [stemmer.stem(word) for word in words]
entire_text = ' '.join(stemmed_words)

negative_cloud.generate(entire_text)

plt.imshow(negative_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
stopwords = set(STOPWORDS)

neutral_cloud = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

entire_text = ' '.join(train_df_neutral['CoronaTweet'].tolist())
words = normal_word_tokenize(entire_text)
stemmed_words = [stemmer.stem(word) for word in words]
entire_text = ' '.join(stemmed_words)

neutral_cloud.generate(entire_text)

plt.imshow(neutral_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
all_words = set()

for sentence in train_df['CoronaTweet']:
    words = normal_word_tokenize(sentence)
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    for word in words:
        all_words.add(word)
        
    if len(words) > 1:
        for i in range(len(words) - 1):
            all_words.add(words[i] + ' ' + words[i + 1])
        
all_words_len = len(all_words)

# %%
train_df_positive = train_df[train_df['Sentiment'] == 'Positive']
positive_wc = {}

sentences = train_df_positive['CoronaTweet']

total_positive_words = 0

for sentence in sentences:
    words = normal_word_tokenize(sentence)
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    # total_positive_words += 2*len(words) - 1
    total_positive_words += len(words)
    
    if len(words) > 1:
        for i in range(len(words) - 1):
            bigram = words[i] + ' ' + words[i + 1]
            if bigram not in positive_wc:
                positive_wc[bigram] = 0
            positive_wc[bigram] += 1  
                
                
    for word in words:
        if word not in positive_wc:
            positive_wc[word] = 0
        positive_wc[word] += 1  

# %%
train_df_negative = train_df[train_df['Sentiment'] == 'Negative']
negative_wc = {}

total_negative_words = 0
sentences = train_df_negative['CoronaTweet']

for sentence in sentences:
    words = normal_word_tokenize(sentence)
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    # total_negative_words += 2*len(words) - 1
    total_negative_words += len(words)
    
    
    if len(words) > 1:
        for i in range(len(words) - 1):
            bigram = words[i] + ' ' + words[i + 1]
            if bigram not in negative_wc:
                negative_wc[bigram] = 0
            negative_wc[bigram] += 1  

    for word in words:
        if word not in negative_wc:
            negative_wc[word] = 0
        negative_wc[word] += 1  

# %%
train_df_neutral = train_df[train_df['Sentiment'] == 'Neutral']
neutral_wc = {}

sentences = train_df_neutral['CoronaTweet']

total_neutral_words = 0


for sentence in sentences:
    words = normal_word_tokenize(sentence)
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    total_neutral_words += 2*len(words) - 1

    
    if len(words) > 1:
        for i in range(len(words) - 1):
            bigram = words[i] + ' ' + words[i + 1]
            if bigram not in neutral_wc:
                neutral_wc[bigram] = 0
            neutral_wc[bigram] += 1  
    
    for word in words:
        if word not in neutral_wc:
            neutral_wc[word] = 0
        neutral_wc[word] += 1  

# %%
# total_positive_words = 0
# for word, count in positive_wc.items():
#     total_positive_words += count
    
# total_negative_words = 0
# for word, count in negative_wc.items():
#     total_negative_words += count
    
# total_neutral_words = 0
# for word, count in neutral_wc.items():
#     total_neutral_words += count

# %%
positive_prob = {}

for word, count in positive_wc.items():
    positive_prob[word] = np.log((count + 1)) - np.log(total_positive_words + all_words_len)
    
negative_prob = {}

for word, count in negative_wc.items():
    negative_prob[word] = np.log((count + 1)) - np.log(total_negative_words + all_words_len)
    
neutral_prob = {}

for word, count in neutral_wc.items():
    neutral_prob[word] = np.log((count + 1)) - np.log(total_neutral_words + all_words_len)
    
# Prior Probabilities, for them laplace smoothing is not that useful
y_is_positive = np.log(len(train_df_positive) / len(train_df))
y_is_negative = np.log(len(train_df_negative) / len(train_df))
y_is_neutral = np.log(len(train_df_neutral) / len(train_df))

# %%
predictions = []
actual = []

for id, row in validate_df.iterrows():
    words = normal_word_tokenize(row['CoronaTweet'])
    
    filtered_words = [word for word in words if word not in stopwords]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    

    positive_sum = y_is_positive
    negative_sum = y_is_negative
    neutral_sum = y_is_neutral
    
    if len(words) > 1:
        for i in range(len(words) - 1):
                bigram = words[i] + ' ' + words[i + 1]
                positive_sum += positive_prob.get(bigram, -np.log(all_words_len + total_positive_words))
                negative_sum += negative_prob.get(bigram, -np.log(all_words_len + total_negative_words))
                neutral_sum += neutral_prob.get(bigram, -np.log(all_words_len + total_neutral_words))
    
    for word in words:
        positive_sum += positive_prob.get(word,-np.log(all_words_len + total_positive_words))
        negative_sum += negative_prob.get(word,  -np.log(all_words_len + total_negative_words))
        neutral_sum += neutral_prob.get(word,  -np.log(all_words_len + total_neutral_words))
    
    if positive_sum >= negative_sum and positive_sum >= neutral_sum:
        predictions.append('Positive')
    elif negative_sum >= positive_sum and negative_sum >= neutral_sum:
        predictions.append('Negative')
    elif neutral_sum >= positive_sum and neutral_sum >= negative_sum:
        predictions.append('Neutral')    
    
    actual.append(row['Sentiment'])


# %%
corrrect = 0
for ac, pred in zip(actual, predictions):
    if ac == pred:
        corrrect += 1
        
print(corrrect / len(actual))

# %%
all_words = set()

for sentence in train_df['CoronaTweet']:
    words = normal_word_tokenize(sentence)
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    for word in words:
        all_words.add(word)
        
    if len(words) > 1:
        for i in range(len(words) - 1):
            all_words.add(words[i] + ' ' + words[i + 1])
        
all_words_len = len(all_words)

# %%
extra_gain_on_hashtag = 1

# %%
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def custom_word_tokeniser(text):
    text = text.lower()
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z#@ \t])|(\w+:\/\/\S+)|^rt|http\S+", "", text)
    # word_tokens =  word_tokenize(text)
    word_tokens = text.split()
    filtered_words = [word for word in word_tokens if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    lematiser = WordNetLemmatizer()
    words = [lematiser.lemmatize(word) for word in words]
    # print(words)
    return words

# %%
all_words = set()

for sentence in train_df['CoronaTweet']:
    words = custom_word_tokeniser(sentence)
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    for word in words:
        all_words.add(word)
        
    # if len(words) > 1:
    #     for i in range(len(words) - 1):
    #         all_words.add(words[i] + ' ' + words[i + 1])
        
all_words_len = len(all_words)

# %%
train_df_positive = train_df[train_df['Sentiment'] == 'Positive']
positive_wc = {}

sentences = train_df_positive['CoronaTweet']

total_positive_words = 0

for sentence in sentences:
    words = custom_word_tokeniser(sentence)
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    # total_positive_words += 2*len(words) - 1
    total_positive_words +=len(words)
    
    # if len(words) > 1:
    #     for i in range(len(words) - 1):
    #         bigram = words[i] + ' ' + words[i + 1]
    #         if bigram not in positive_wc:
    #             positive_wc[bigram] = 0
    #         positive_wc[bigram] += 1  
                
                
    for word in words:
        if word not in positive_wc:
            positive_wc[word] = 0
        positive_wc[word] += 1  
    
    for word in words:
        if '#' in word:
            positive_wc[word] += extra_gain_on_hashtag
            total_positive_words += extra_gain_on_hashtag

# %%
train_df_negative = train_df[train_df['Sentiment'] == 'Negative']
negative_wc = {}

total_negative_words = 0
sentences = train_df_negative['CoronaTweet']

for sentence in sentences:
    words = custom_word_tokeniser(sentence)
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    
    # total_negative_words += 2*len(words) - 1
    total_negative_words += len(words)
    
    
    # if len(words) > 1:
    #     for i in range(len(words) - 1):
    #         bigram = words[i] + ' ' + words[i + 1]
    #         if bigram not in negative_wc:
    #             negative_wc[bigram] = 0
    #         negative_wc[bigram] += 1  

    for word in words:
        if word not in negative_wc:
            negative_wc[word] = 0
        negative_wc[word] += 1  
    
    for word in words:
        if '#' in word:
            negative_wc[word] += extra_gain_on_hashtag
            total_negative_words += extra_gain_on_hashtag

# %%
train_df_neutral = train_df[train_df['Sentiment'] == 'Neutral']
neutral_wc = {}

sentences = train_df_neutral['CoronaTweet']

total_neutral_words = 0


for sentence in sentences:
    words = custom_word_tokeniser(sentence)
    
    filtered_words = [word for word in words if word not in stopwords]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    # total_neutral_words += 2*len(words) - 1
    total_neutral_words += len(words)

    
    # if len(words) > 1:
    #     for i in range(len(words) - 1):
    #         bigram = words[i] + ' ' + words[i + 1]
    #         if bigram not in neutral_wc:
    #             neutral_wc[bigram] = 0
    #         neutral_wc[bigram] += 1  
    
    for word in words:
        if word not in neutral_wc:
            neutral_wc[word] = 0
        neutral_wc[word] += 1  
    
    for word in words:
        if '#' in word:
            neutral_wc[word] += extra_gain_on_hashtag
            total_neutral_words += extra_gain_on_hashtag

# %%
positive_prob = {}

for word, count in positive_wc.items():
    positive_prob[word] = np.log((count + 1)) - np.log(total_positive_words + all_words_len)
    
negative_prob = {}

for word, count in negative_wc.items():
    negative_prob[word] = np.log((count + 1)) - np.log(total_negative_words + all_words_len)
    
neutral_prob = {}

for word, count in neutral_wc.items():
    neutral_prob[word] = np.log((count + 1)) - np.log(total_neutral_words + all_words_len)
    
# Prior Probabilities, for them laplace smoothing is not that useful
y_is_positive = np.log(len(train_df_positive) / len(train_df))
y_is_negative = np.log(len(train_df_negative) / len(train_df))
y_is_neutral = np.log(len(train_df_neutral) / len(train_df))

# %%
predictions = []
actual = []

for id, row in validate_df.iterrows():
    words = custom_word_tokeniser(row['CoronaTweet'])
    
    filtered_words = [word for word in words if word not in stopwords]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in filtered_words]
    

    positive_sum = y_is_positive
    negative_sum = y_is_negative
    neutral_sum = y_is_neutral
    
    # if len(words) > 1:
    #     for i in range(len(words) - 1):
    #             bigram = words[i] + ' ' + words[i + 1]
    #             positive_sum += positive_prob.get(bigram, -np.log(all_words_len + total_positive_words))
    #             negative_sum += negative_prob.get(bigram, -np.log(all_words_len + total_negative_words))
    #             neutral_sum += neutral_prob.get(bigram, -np.log(all_words_len + total_neutral_words))
    
    for word in words:
        positive_sum += positive_prob.get(word,-np.log(all_words_len + total_positive_words))
        negative_sum += negative_prob.get(word,  -np.log(all_words_len + total_negative_words))
        neutral_sum += neutral_prob.get(word,  -np.log(all_words_len + total_neutral_words))
    
    if positive_sum >= negative_sum and positive_sum >= neutral_sum:
        predictions.append('Positive')
    elif negative_sum >= positive_sum and negative_sum >= neutral_sum:
        predictions.append('Negative')
    elif neutral_sum >= positive_sum and neutral_sum >= negative_sum:
        predictions.append('Neutral')    
    
    actual.append(row['Sentiment'])


# %%
corrrect = 0
for ac, pred in zip(actual, predictions):
    if ac == pred:
        corrrect += 1
        
print(corrrect / len(actual))

# %%
# stop()

# %%
all_accuracy = [] 

for i in [1,2,5,10,25,50,100]:
    extra_data = pd.read_csv(f'./Domain_Adaptation/Twitter_train_{i}.csv')
    extra_data = extra_data.rename(columns={'Tweet': 'CoronaTweet'})

    all_input_data = pd.concat([train_df, extra_data])
    all_words = set()

    for sentence in all_input_data['CoronaTweet']:
        words = custom_word_tokeniser(sentence)
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        
        for word in words:
            all_words.add(word)
            
    all_words_len = len(all_words)
    
    train_df_positive = all_input_data[all_input_data['Sentiment'] == 'Positive']
    positive_wc = {}

    total_positive_words = 0

    sentences = train_df_positive['CoronaTweet']

    for sentence in sentences:
        words = custom_word_tokeniser(sentence)
        
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        
        total_positive_words += len(words)
        
        for word in words:
            if word not in positive_wc:
                positive_wc[word] = 0
            positive_wc[word] += 1  
    
    train_df_negative = all_input_data[all_input_data['Sentiment'] == 'Negative']
    negative_wc = {}

    sentences = train_df_negative['CoronaTweet']
    total_negative_words = 0

    for sentence in sentences:
        words = custom_word_tokeniser(sentence)
        
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        total_negative_words += len(words)
        
        
        for word in words:
            if word not in negative_wc:
                negative_wc[word] = 0
            negative_wc[word] += 1  
    
    train_df_neutral = all_input_data[all_input_data['Sentiment'] == 'Neutral']
    neutral_wc = {}

    sentences = train_df_neutral['CoronaTweet']
    total_neutral_words = 0


    for sentence in sentences:
        words = custom_word_tokeniser(sentence)
        
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        total_neutral_words += len(words)
        
        
        for word in words:
            if word not in neutral_wc:
                neutral_wc[word] = 0
            neutral_wc[word] += 1  
    
    positive_prob = {}

    for word, count in positive_wc.items():
        positive_prob[word] = np.log((count + 1)) - np.log(total_positive_words + all_words_len)
        
    negative_prob = {}

    for word, count in negative_wc.items():
        negative_prob[word] = np.log((count + 1)) - np.log(total_negative_words + all_words_len)
        
    neutral_prob = {}

    for word, count in neutral_wc.items():
        neutral_prob[word] = np.log((count + 1)) - np.log(total_neutral_words + all_words_len)  
        
    # Prior Probabilities, for them laplace smoothing is not that useful
    y_is_positive = np.log(len(train_df_positive) / len(all_input_data))
    y_is_negative = np.log(len(train_df_negative) / len(all_input_data))
    y_is_neutral = np.log(len(train_df_neutral) / len(all_input_data))


    new_validate_df = pd.read_csv('./Domain_Adaptation/Twitter_validation.csv')
    new_validate_df = new_validate_df.rename(columns={'Tweet': 'CoronaTweet'})

    predictions = []
    actual = []
    for id, row in new_validate_df.iterrows():
        
        words = custom_word_tokeniser(row['CoronaTweet'])
        validate_df
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        
        positive_sum = y_is_positive
        negative_sum = y_is_negative
        neutral_sum = y_is_neutral
        
        for word in words:
            if word not in all_words:
                continue
            positive_sum += positive_prob.get(word,-np.log(all_words_len + total_positive_words))
            negative_sum += negative_prob.get(word,  -np.log(all_words_len + total_negative_words))
            neutral_sum += neutral_prob.get(word,  -np.log(all_words_len + total_neutral_words))
        
        if positive_sum >= negative_sum and positive_sum >= neutral_sum:
            predictions.append('Positive')
        elif negative_sum >= positive_sum and negative_sum >= neutral_sum:
            predictions.append('Negative')
        elif neutral_sum >= positive_sum and neutral_sum >= negative_sum:
            predictions.append('Neutral')    
        
        actual.append(row['Sentiment'])
        
    corrrect = 0
    for ac, pred in zip(actual, predictions):
        if ac == pred:
            corrrect += 1
            
    print(f'For {i} extra data, accuracy is {corrrect / len(actual)}')
    all_accuracy.append(corrrect / len(actual))


# %%
all_accuracy_no_source = [] 

for i in [1,2,5,10,25,50,100]:
    all_input_data = pd.read_csv(f'./Domain_Adaptation/Twitter_train_{i}.csv')
    all_input_data = all_input_data.rename(columns={'Tweet': 'CoronaTweet'})
    # all_input_data = pd.concat([train_df, extra_data])
    all_words = set()

    for sentence in all_input_data['CoronaTweet']:
        words = custom_word_tokeniser(sentence)
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        
        for word in words:
            all_words.add(word)
            
    all_words_len = len(all_words)
    
    train_df_positive = all_input_data[all_input_data['Sentiment'] == 'Positive']
    positive_wc = {}

    total_positive_words = 0

    sentences = train_df_positive['CoronaTweet']

    for sentence in sentences:
        words = custom_word_tokeniser(sentence)
        
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        
        total_positive_words += len(words)
        
        for word in words:
            if word not in positive_wc:
                positive_wc[word] = 0
            positive_wc[word] += 1  
    
    train_df_negative = all_input_data[all_input_data['Sentiment'] == 'Negative']
    negative_wc = {}

    sentences = train_df_negative['CoronaTweet']
    total_negative_words = 0

    for sentence in sentences:
        words = custom_word_tokeniser(sentence)
        
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        total_negative_words += len(words)
        
        
        for word in words:
            if word not in negative_wc:
                negative_wc[word] = 0
            negative_wc[word] += 1  
    
    train_df_neutral = all_input_data[all_input_data['Sentiment'] == 'Neutral']
    neutral_wc = {}

    sentences = train_df_neutral['CoronaTweet']
    total_neutral_words = 0


    for sentence in sentences:
        words = custom_word_tokeniser(sentence)
        
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        total_neutral_words += len(words)
        
        
        for word in words:
            if word not in neutral_wc:
                neutral_wc[word] = 0
            neutral_wc[word] += 1  
    
    positive_prob = {}

    for word, count in positive_wc.items():
        positive_prob[word] = np.log((count + 1)) - np.log(total_positive_words + all_words_len)
        
    negative_prob = {}

    for word, count in negative_wc.items():
        negative_prob[word] = np.log((count + 1)) - np.log(total_negative_words + all_words_len)
        
    neutral_prob = {}

    for word, count in neutral_wc.items():
        neutral_prob[word] = np.log((count + 1)) - np.log(total_neutral_words + all_words_len)  
        
    # Prior Probabilities, for them laplace smoothing is not that useful
    y_is_positive = np.log(len(train_df_positive) / len(all_input_data))
    y_is_negative = np.log(len(train_df_negative) / len(all_input_data))
    y_is_neutral = np.log(len(train_df_neutral) / len(all_input_data))


    new_validate_df = pd.read_csv('./Domain_Adaptation/Twitter_validation.csv')
    new_validate_df = new_validate_df.rename(columns={'Tweet': 'CoronaTweet'})
    

    predictions = []
    actual = []
    for id, row in new_validate_df.iterrows():
        
        words = custom_word_tokeniser(row['CoronaTweet'])
        validate_df
        filtered_words = [word for word in words if word not in stopwords]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in filtered_words]
        
        positive_sum = y_is_positive
        negative_sum = y_is_negative
        neutral_sum = y_is_neutral
        
        for word in words:
            if word not in all_words:
                continue
            positive_sum += positive_prob.get(word,-np.log(all_words_len + total_positive_words))
            negative_sum += negative_prob.get(word,  -np.log(all_words_len + total_negative_words))
            neutral_sum += neutral_prob.get(word,  -np.log(all_words_len + total_neutral_words))
        
        if positive_sum >= negative_sum and positive_sum >= neutral_sum:
            predictions.append('Positive')
        elif negative_sum >= positive_sum and negative_sum >= neutral_sum:
            predictions.append('Negative')
        elif neutral_sum >= positive_sum and neutral_sum >= negative_sum:
            predictions.append('Neutral')    
        
        actual.append(row['Sentiment'])
        
    corrrect = 0
    for ac, pred in zip(actual, predictions):
        if ac == pred:
            corrrect += 1
            
    print(f'For {i} extra data, accuracy is {corrrect / len(actual)}')
    all_accuracy_no_source.append(corrrect / len(actual))


# %%
fig, ax = plt.subplots()
x_axis_at = [0,1,2,3,4,5,6]
percentages = [1, 2, 5, 10, 25, 50, 100]

ax.plot(x_axis_at, [100*x for x in all_accuracy], label='With data from source', marker='o')
ax.plot(x_axis_at, [100*x for x in all_accuracy_no_source], label='Without data from source', marker='s')

ax.set_xlabel('Percentage of Data')
ax.set_ylabel('Accuracy')
ax.set_title('Comparing injecting source data')

ax.set_xticks(x_axis_at)
ax.set_xticklabels([str(p) + '%' for p in percentages])

ax.legend()

plt.grid(True)
plt.show()


