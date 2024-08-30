import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')

df = pd.read_csv('./SQLi_TrainingData.csv', usecols=[0,1])
X = df['Sentence']
Y= df['Label']

vectorizer = CountVectorizer(min_df=2, max_df=0.8, stop_words= stopwords.words('english'))
X = vectorizer.fit_transform(X.values.astype('U')).toarray()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)
y_pred = nb_clf.predict(X_test)

print(f"Accuracy of Gaussian Naive Bayes on test set : {accuracy_score(y_pred, Y_test)}")
#print(f"F1 Score of Gaussian Naive Bayes on test set: {f1_score(y_pred, Y_test)}")

confusion_matrix = confusion_matrix(Y_test, y_pred)
TP = confusion_matrix[1, 1]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]

sensitivity = TP/float(FN+TP)
print("Sensitivity = ", sensitivity)

specificity = TN/float(FP+TN)
print("Specificity = ", specificity)

precision = TP/float(TP+FP)
recall = TP/float(TP+FN)
F1 = 2*((precision*recall)/(precision+recall))
print("Percision = ", precision)

#df.hist()
#pyplot.show()

# TODO
# -Create Data Visualizations
# -Determine how to interpret user input
# -Create UI