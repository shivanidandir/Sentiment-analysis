
# coding: utf-8

# In[5]:


import tweepy,json
access_token=""
access_token_secret=""
consumer_key=""
consumer_secret=""
auth= tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)


# In[6]:


import tweepy
tweet_list=[]
class MyStreamListener(tweepy.StreamListener):
    def __init__(self,api=None):
        super(MyStreamListener,self).__init__()
        self.num_tweets=0
        self.file=open("test.txt","w")
    def on_status(self,status):
        tweet=status._json
        self.file.write(json.dumps(tweet)+ '\n')
        tweet_list.append(status)
        self.num_tweets+=1
        if self.num_tweets<1000:
            return True
        else:
            return False
        self.file.close()


# In[7]:


#create streaming object and authenticate
l = MyStreamListener()
stream =tweepy.Stream(auth,l)
#this line filters twiiter streams to capture data by keywords
stream.filter(track=['trump','racism','racist','women','people','support','abuse','power','video','sexist','sexism','misogyny'])


# In[8]:


import json
tweets_data_path='test.txt'
tweets_data=[]
tweets_file=open(tweets_data_path,"r")
#read in tweets and store on list
for line in tweets_file:
    tweet=json.loads(line)
    tweets_data.append(tweet)
tweets_file.close()
print(tweets_data[0])


# In[9]:


import numpy as np
import pandas as pd


# In[10]:


train  = pd.read_csv('\train_E6oV3lV.csv')
train.head()


# In[11]:


train.info()


# In[12]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, accuracy_score


# In[13]:


train.label.value_counts()


# In[14]:



import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[15]:


stop_words=[]
stop_words = set(stopwords.words('english'))
stop = [x.lower() for x in stop_words]
lemma = WordNetLemmatizer()

shortcuts = {'u': 'you', 'y': 'why', 'r': 'are', 'doin': 'doing', 'hw': 'how', 'k': 'okay', 'm': 'am', 'b4': 'before',
            'idc': "i do not care", 'ty': 'thankyou', 'wlcm': 'welcome', 'bc': 'because', '<3': 'love', 'xoxo': 'love',
            'ttyl': 'talk to you later', 'gr8': 'great', 'bday': 'birthday', 'awsm': 'awesome', 'gud': 'good', 'h8': 'hate',
            'lv': 'love', 'dm': 'direct message', 'rt': 'retweet', 'wtf': 'hate', 'idgaf': 'hate',
             'irl': 'in real life', 'yolo': 'you only live once'}

def clean(text):
    text = text.lower()
    # keep alphanumeric characters only
    text = re.sub('\W+', ' ', text).strip()
    text = text.replace('user', '')
    # tokenize
    text_token = word_tokenize(text)
    # replace shortcuts using dict
    full_words = []
    for token in text_token:
        if token in shortcuts.keys():
            token = shortcuts[token]
        full_words.append(token)
#     text = " ".join(full_words)
#     text_token = word_tokenize(text)
    # stopwords removal
#     words = [word for word in full_words if word not in stop]
    words_alpha = [re.sub(r'\d+', '', word) for word in full_words]
    words_big = [word for word in words_alpha if len(word)>2]
    stemmed_words = [lemma.lemmatize(word) for word in words_big]
    # join list elements to string
    clean_text = " ".join(stemmed_words)
    clean_text = clean_text.replace('   ', ' ')
    clean_text = clean_text.replace('  ', ' ')
    return clean_text


# In[16]:


hypocrite = []
for i in range(len(train['tweet'])):
    if 'hypocrite' in train['tweet'][i]:
        if train['label'][i] == 1:
            hypocrite.append('racist')
        else:
            hypocrite.append('good')
    else:
        hypocrite.append('good')
df = pd.DataFrame(columns=['hypocrite'], data=hypocrite)
print(df['hypocrite'].value_counts())

train['hypocrite'] = hypocrite


# In[17]:


train['combined'] = train['tweet'].apply(str) + ' ' + train['hypocrite'].apply(str)


# In[18]:


X_train = train.combined
y = train.label


# In[26]:


test= pd.DataFrame(tweets_data,columns=['text'])
test.head()


# In[28]:


test.columns=['tweet']
test.head()


# In[29]:


X_test = test.tweet


# In[30]:


clean_Xtrain = X_train.apply(lambda x: clean(x))


# In[31]:


clean_Xtrain[1531]


# In[32]:



clean_Xtest = X_test.apply(lambda x: clean(x))


# In[33]:



print(len(clean_Xtrain))
print(len(clean_Xtest))
print(len(y))


# In[34]:



vectorizer = CountVectorizer(max_df=0.5)
# vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.5)

X = vectorizer.fit_transform(clean_Xtrain)
X_test = vectorizer.transform(clean_Xtest)


# In[35]:


print(X.shape)
print(X_test.shape)


# In[36]:


model = LinearSVC(penalty='l2', C=0.5, dual=False, random_state=0, max_iter=1000)
print(model)


# In[37]:


# split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)

# calculate f1 score
model.fit(X_train,y_train)
y_pred = model.predict(X_val)
print('Accuracy:', accuracy_score(y_pred, y_val))
print("F1 Score: ", f1_score(y_pred, y_val))


# In[38]:



df = pd.DataFrame()
df['y_pred'] = y_pred
df['y_pred'].value_counts()


# In[39]:


# train model with full data and predict for new samples
model.fit(X, y)
y_pred = model.predict(X_test)


# In[40]:



df = pd.DataFrame()
df['y_pred'] = y_pred
df['y_pred'].value_counts()


# In[42]:


all_words = ' '.join([text for text in train['tweet']])
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,background_color='white',stopwords=STOPWORDS).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[45]:


normal_words =' '.join([text for text in train['tweet'][train['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,background_color='white',stopwords=STOPWORDS).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[46]:


negative_words = ' '.join([text for text in train['tweet'][train['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,random_state=21, max_font_size=110,background_color='white',stopwords=STOPWORDS).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

