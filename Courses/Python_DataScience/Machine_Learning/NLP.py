# Natural Language Processing

"""
import nltk

# opens an interactive menu
#nltk.download_shell()



#messages = [line.rstrip() for line in open("SMSSpamCollection")]
#print(messages)

"""
""""
for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
"""
"""

import pandas as pd

messages = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])

#print(messages.head())
#print(messages.describe())

#print(messages.groupby('label').describe())

messages['length'] = messages['message'].apply(len)

#print(messages.head())

import matplotlib.pyplot as plt
import seaborn as sb

#messages['length'].plot.hist(bins=50,edgecolor='black')
#plt.show()

#print(messages[messages['length'] == 910]['message'].iloc[0])

#messages.hist(column='length', by='label', bins=60, figsize=(12,4))
#plt.show()

import string

mess = 'Sample Message! Notice: it has punctuation.'

#print(string.punctuation)

nopunc = [c for c in mess if c not in string.punctuation]

#print(nopunc)

from nltk.corpus import stopwords

#stopwords.words('english')

nopunc = ''.join(nopunc)

nopunc.split()

print(nopunc)

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#print(clean_mess)

def text_process(mess):
"""
"""
1. remove punc
2. remove stop words
3. return list of clean text words
"""
"""

    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#print(messages['message'].head(5).apply(text_process))

from sklearn.feature_extraction.text import CountVectorizer

# bag of words
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

#print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]
#print(mess4)

bow4 = bow_transformer.transform([mess4])
#print(bow4)

#bow_transformer.get_feature_names()[4068]

message_bow = bow_transformer.transform(messages['message'])

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(message_bow)

tfidf4 = tfidf_transformer.transform(bow4)

print(tfidf4)

tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

messages_tfidf = tfidf_transformer.transform((message_bow))

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

spam_detect_model.predict(tfidf4)

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'],test_size=0.3)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report

print(classification_report(label_test, predictions))

"""

# NLP Project

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


yelp = pd.read_csv('yelp.csv')

#print(yelp.head())
#print(yelp.info())
#rint(yelp.describe())


yelp['text length'] = (yelp['text'].apply(lambda x: len(x.split())))

#print(yelp.head())


#g = sb.FacetGrid(yelp, col="stars")
#g = g.map(plt.hist, 'text length', edgecolor='black')

#sb.boxplot(x=yelp['stars'],y=yelp['text length'])

#sb.countplot(yelp['stars'])
#sb.plt.show()

grouped = yelp.groupby('stars')

#print(grouped.mean())
#print(grouped.mean().corr())

grouped_cor = grouped.mean().corr()

#sb.heatmap(grouped_cor)
#sb.plt.show()

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
#print(yelp_class)

X = yelp_class['text']
y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=101)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train, y_train)

predictions = nb.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

X = yelp_class['text']
y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


"""
predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report

print(classification_report(label_test, predictions))

"""

