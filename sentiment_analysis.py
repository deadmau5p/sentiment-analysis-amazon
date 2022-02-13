import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from keras import Sequential
from keras.layers import LSTM


reviews_data = pd.read_csv("./1429_1.csv")
#take only these features
reviews_data = reviews_data[['reviews.rating' , 'reviews.text']]
#filtering null ratings
reviews_data = reviews_data[reviews_data['reviews.rating'].notnull()]
reviews_data["reviews.rating"] = reviews_data["reviews.rating"].astype(int)
#reviews_data['reviews.rating'].value_counts().plot.bar()
#plt.show()
stop_words = stopwords.words('english')
porter = PorterStemmer()

def text_clean(text):
    text = str(text).lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    stemmed = [porter.stem(word) for word in words]
    return stemmed

#apply text cleaning to every review
reviews_data['reviews.text'] = reviews_data['reviews.text'].apply(text_clean)
#we split the data in train and test
train_data, test_data = train_test_split(reviews_data, test_size=0.3)

train_data['reviews.text'] = train_data['reviews.text'].apply(lambda list: ' '.join(list))
test_data['reviews.text'] = test_data['reviews.text'].apply(lambda list: ' '.join(list))

vectorizer = CountVectorizer()
vectors_train = vectorizer.fit_transform(train_data['reviews.text'])
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(vectors_train)

vectors_test = vectorizer.transform(test_data['reviews.text'])
x_test_tfidf = tfidf_transformer.transform(vectors_test)

model = SVC(verbose=True)
model.fit(x_train_tfidf, train_data['reviews.rating'])


#task of predicting rating
prediction = model.predict(x_test_tfidf)
print('The accuracy of the SVM is:', metrics.accuracy_score(prediction,test_data['reviews.rating']))

#task of predicting of positive or negative review
prediction1 = np.where(prediction >= 4, "pos", "neg")
test_y = test_data['reviews.rating'].apply(lambda x: "pos" if x >= 4 else "neg")
print('The accuracy of the SVM is:', metrics.accuracy_score(prediction1,test_y))

model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train_tfidf, train_data['reviews.rating'])

#task of predicting rating
prediction = model.predict(x_test_tfidf)
print('The accuracy of the KNN is:', metrics.accuracy_score(prediction,test_data['reviews.rating']))

#task of predicting of positive or negative review
prediction1 = np.where(prediction >= 4, "pos", "neg")
test_y = test_data['reviews.rating'].apply(lambda x: "pos" if x >= 4 else "neg")
print('The accuracy of the KNN is:', metrics.accuracy_score(prediction1,test_y))

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_tfidf, train_data['reviews.rating'])

prediction = model.predict(x_test_tfidf)
print('The accuracy of the KNN is:', metrics.accuracy_score(prediction,test_data['reviews.rating']))
prediction1 = np.where(prediction >= 4, "pos", "neg")
test_y = test_data['reviews.rating'].apply(lambda x: "pos" if x >= 4 else "neg")
print('The accuracy of the KNN is:', metrics.accuracy_score(prediction1,test_y))







