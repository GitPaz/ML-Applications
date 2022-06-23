# Email Classification Using ML Project
# James Krepps
# U34951440
# Objective is to design a spam filter using a ML model. I have chosen logistic
# regresssion for this program. Specifically, binary logistic regression 
# because it allows for a simple yes or no determinations such as spam or not
# spam. 
# A tremendous thanks to authors Avinash Navlani and Mr. Unity Buddy. 
# Your individual documentations have been a lifesaver.

###############################################################################
                     ####GNU General Public License####
#End users have the freedom to run, study, share, and modify this software as
#                             they see fit.
###############################################################################

import pandas as pd
# pandas is a science and data package for python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 
# science kit learn is a package containing science and data tools for python

path = "C:/Users/Admin/Desktop/Spam Project/hamnspam"
hamPath = "C:/Users/Admin/Desktop/Spam Project/hamnspam/ham"
spamPath = "C:/Users/Admin/Desktop/Spam Project/hamnspam/spam"

columns = ['v1','v2']
# assign var columns with v1 and v2 to define column bounds

data = pd.read_csv(r'C:\Users\Admin\Desktop\SpamProject\Spam.csv', 
                   encoding="ISO-8859-1")
# read in spam file and assign to data var

df = pd.DataFrame(data, columns= ['v1','v2'])
# creates a data frame built off of column conditions v1 and v2. Essentially,
# creating a 2-dimensional, labeled, data structure.

# df.v2.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
# consider using word frequency as a means to identify phishing schemes.

freq = df.groupby(['v2']).size()
display(freq)
# this works, kind of.

print(df)
# verify that the data is being imported properly

#def stringify(text):
#    removepunct = "".join([char for char in text if char not in string.punctuation])
#    return removepunct
#def token_conversion(text):
#    spam_tokens = re.split('\W+', text)
#    return spam_tokens

x = df['v2']
# set x equal to the data contained in v2
y = df['v1']
# set y = to the label assigned to v1, describing v2

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.93)
# takes x and y, our columns v1 and v2, and sets 25% as the testing sample
# and the remainder as the training sample. Accuracy remain above 90% for all
# N < 0.94

Tokenized = CountVectorizer()
# CountVectorizer tokenizeseach word, saves the number of occurrences, and
# then saves that result to the var occ.
# Format is: email ID, word ID, Occurrence
features = Tokenized.fit_transform(x_train)

print(features)

# With a classification system in place, the next step will be to model the
# data and plot it in a way that's useful. 

plot = svm.SVC()
# Support Vector Machine is used to plot data points around linear line in order to judge accuracy. 
# Consider possibly using random forest method in future.
plot.fit(features,y_train)

features_test = Tokenized.transform(x_test)
print("Current accuracy: {}".format(plot.score(features_test,y_test)))

"""
Current Testing List

# SPAM Alternative
# Natural Language Processing Variant

import nltk 
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import texthero as hero
import pandas as pd
import string
import re
import nltk
import cleantext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm 
from sklearn import datasets 
from cleantext import clean
string.punctuation
stopword = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()


def stringify(text):
    removepunct = "".join([char for char in text if char not in string.punctuation])
    return removepunct

def token_conversion(text):
    spam_tokens = re.split('\W+', text)
    return spam_tokens

def simplify(tokenfy_list):
    simplified_text = [word for word in tokenfy_list if word not in stopword]
    return simplified_text 

def lemmatizing(tokens):
    text = [wn.lemmatize(word) for word in tokens]
    return text


columns = ['v1','v2']    
df = pd.read_csv(r'C:\Users\Admin\Desktop\SpamProject\Spam.csv', encoding="ISO-8859-1")
print(df)

df['v3'] =  df['v2'].apply(lambda x: stringify(x))
print(df)

df['v4'] = df['v3'].apply(lambda x: token_conversion(x.lower()))
print(df)

df['v5'] = df['v4'].apply(lambda x: simplify(x))
print(df)

df['v6'] = df['v5'].apply(lambda x: lemmatizing(x))
print(df)


count_vect = CountVectorizer(analyzer=hero.clean)
X_counts = count_vect.fit_transform(df['v6'])
print(X_counts.shape)
print(count_vect.get_feature_names_out())

"""
