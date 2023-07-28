# fake-news-detection
fake news detection
In [1]:
In [2]:
inserting a column "class" as target feature
In [3]:
In [4]:
Removing last 10 rows for manual testing
In [5]:
In [6]:
In [7]:
Out[4]:
((21417, 5), (23481, 5))
Out[7]:
((23481, 5), (21417, 5))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
df_fake=pd.read_csv("Fake.csv")
df_true=pd.read_csv("True.csv")
df_fake["class"] = 0
df_true["class"] = 1
df_true.shape,df_fake.shape
df_fake_manual_testing=df_fake.tail(10)
for i in range(21417,23481,-1):
df_fake.drop([i],axis=0,inplace=True)
df_true_manual_testing=df_true.tail(10)
for i in range(21417,23481,-1):
df_true.drop([i],axis=0,inplace=True)
df_fake.shape,df_true.shape
In [8]:
In [9]:
C:\Users\user\AppData\Local\Temp\ipykernel_13112\401406962.py:1: Setting
WithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead
See the caveats in the documentation: https://pandas.pydata.org/pandas-d
ocs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy (http
s://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#return
ing-a-view-versus-a-copy)
df_fake_manual_testing["class"]=0
C:\Users\user\AppData\Local\Temp\ipykernel_13112\401406962.py:2: Setting
WithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead
See the caveats in the documentation: https://pandas.pydata.org/pandas-d
ocs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy (http
s://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#return
ing-a-view-versus-a-copy)
df_true_manual_testing["class"]=1
Out[9]:
title text subject date class
23471
Seven Iranians freed in the
prisoner swap have...
21st Century Wire says This
week, the historic...
Middleeast
January
20, 2016
0
23472
#Hashtag Hell & The Fake
Left
By Dady Chery and Gilbert
MercierAll writers ...
Middleeast
January
19, 2016
0
23473
Astroturfing: Journalist
Reveals Brainwashing ...
Vic Bishop Waking TimesOur
reality is carefull...
Middleeast
January
19, 2016
0
23474
The New American
Century:An Era of Fraud
Paul Craig RobertsIn the last
years of the 20t...
Middleeast
January
19, 2016
0
23475
Hillary Clinton: ‘Israel First’
(and no peace ...
Robert Fantina
CounterpunchAlthough the
United...
Middleeast
January
18, 2016
0
23476
McPain: John McCain
Furious That Iran Treated
...
21st Century Wire says As
21WIRE reported earl...
Middleeast
January
16, 2016
0
23477
JUSTICE? Yahoo Settles
E-mail Privacy Class-ac...
21st Century Wire says It s a
familiar theme. ...
Middleeast
January
16, 2016
0
23478
Sunnistan: US and Allied
‘Safe Zone’ Plan to T...
Patrick Henningsen 21st Century
WireRemember ...
Middleeast
January
15, 2016
0
23479
How to Blow $700 Million:
Al Jazeera America F...
21st Century Wire says Al
Jazeera America will...
Middleeast
January
14, 2016
0
23480
10 U.S. Navy Sailors Held
by Iranian Military ...
21st Century Wire says As
21WIRE predicted in ...
Middleeast
January
12, 2016
0
df_fake_manual_testing["class"]=0
df_true_manual_testing["class"]=1
df_fake_manual_testing.head(10)
In [10]:
creating like pandas
In [11]:
Out[10]:
title text subject date class
21407
Mata Pires, owner of
embattled Brazil builder ...
SAO PAULO (Reuters) - Cesar
Mata Pires, the ow...
worldnews
August 22,
2017
1
21408
U.S., North Korea clash at
U.N. forum over nuc...
GENEVA(Reuters) - North
Korea and the United ...
worldnews
August 22,
2017
1
21409
U.S., North Korea clash at
U.N. arms forum on ...
GENEVA(Reuters) - North
Korea and the United ...
worldnews
August 22,
2017
1
21410
Headless torso could
belong to submarine
journ...
COPENHAGEN (Reuters) -
Danish police said on T...
worldnews
August 22,
2017
1
21411
North Korea shipments to
Syria chemical arms a...
UNITED NATIONS (Reuters) -
Two North Korean sh...
worldnews
August 21,
2017
1
21412
'Fully committed' NATO
backs new U.S. approach...
BRUSSELS (Reuters) - NATO
allies on Tuesday we...
worldnews
August 22,
2017
1
21413
LexisNexis withdrew two
products from Chinese ...
LONDON (Reuters) -
LexisNexis, a provider of l...
worldnews
August 22,
2017
1
21414
Minsk cultural hub
becomes haven from
authorities
MINSK (Reuters) - In the
shadow of disused Sov...
worldnews
August 22,
2017
1
21415
Vatican upbeat on
possibility of Pope Francis
...
MOSCOW (Reuters) - Vatican
Secretary of State ...
worldnews
August 22,
2017
1
21416
Indonesia to buy $1.14
billion worth of Russia...
JAKARTA(Reuters) - Indonesia
will buy 11 Sukh...
worldnews
August 22,
2017
1
df_true_manual_testing.head(10)
df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis =
df_manual_testing.to_csv("manual_testing.csv")
In [12]:
In [13]:
In [14]:
In [15]:
Random Shuffling the dataframe
Out[12]:
title text subject date class
0
Donald Trump Sends Out
Embarrassing New Year’...
Donald Trump just couldn t
wish all Americans ...
News
December 31,
2017
0
1
Drunk Bragging Trump Staffer
Started Russian ...
House Intelligence
Committee Chairman Devin
Nu...
News
December 31,
2017
0
2
Sheriff David Clarke Becomes An
Internet Joke...
On Friday, it was revealed that
former Milwauk...
News
December 30,
2017
0
3
Trump Is So Obsessed He Even
Has Obama’s Name...
On Christmas day, Donald
Trump announced that ...
News
December 29,
2017
0
4
Pope Francis Just Called Out
Donald Trump Dur...
Pope Francis used his annual
Christmas Day mes...
News
December 25,
2017
0
5
RacistAlabama Cops Brutalize
Black Boy While...
The number of cases of cops
brutalizing and ki...
News
December 25,
2017
0
6
Fresh Off The Golf Course,
Trump Lashes OutA...
Donald Trump spent a good
portion of his day a...
News
December 23,
2017
0
7
Trump Said Some INSANELY
Racist Stuff Inside ...
In the wake of yet another
court decision that...
News
December 23,
2017
0
8
Former CIADirector Slams
Trump Over UN Bully...
Many people have raised the
alarm regarding th...
News
December 22,
2017
0
9
WATCH: Brand-New Pro-Trump
Ad Features So Muc...
Just when you might have
thought we d get a br...
News
December 21,
2017
0
Out[13]:
Index(['title', 'text', 'subject', 'date', 'class'], dtype='object')
Out[15]:
text 0
class 0
dtype: int64
df_merge=pd.concat([df_fake,df_true],axis=0)
df_merge.head(10)
df_merge.columns
df =df_merge.drop(["title","subject","date"],axis=1)
df.isnull().sum()
In [16]:
In [17]:
In [18]:
In [19]:
In [20]:
Creating a function to process the texts
Out[16]:
text class
0 Donald Trump just couldn t wish all Americans ... 0
1 House Intelligence Committee Chairman Devin Nu... 0
2 On Friday, it was revealed that former Milwauk... 0
3 On Christmas day, Donald Trump announced that ... 0
4 Pope Francis used his annual Christmas Day mes... 0
Out[19]:
Index(['text', 'class'], dtype='object')
Out[20]:
text class
0 (Reuters) - U.S. President Donald Trump’s nomi... 1
1 BAGHDAD/ERBIL, Iraq (Reuters) - Iraq s cabinet... 1
2 As Donald Trump continues to take criticism fr... 0
3 With everything going on at Donald Trump ralli... 0
4 WASHINGTON (Reuters) - U.S. President Barack O... 1
df.head()
df=df.sample(frac = 1)
df.reset_index(inplace= True)
df.drop(["index"],axis =1,inplace=True)
df.columns
df.head()
In [*]:
In [*]:
Defining dependent and independent variables
In [*]:
Splitting Training and Testing
In [*]:
convert text to vectors
In [*]:
Logistic Regression
In [*]:
def wordopt(text):
text = text.lower()
text = re.sub('\[.*?\]', '', text)
text = re.sub("\\W"," ",text)
text = re.sub('https?://\S+|www\.\S+', '', text)
text = re.sub('<.*?>+', '', text)
text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
text = re.sub('\n', '', text)
text = re.sub('\w*\d\w*', '', text)
return text
df["text"]=df["text"].apply(wordopt)
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train,y_train)
In [*]:
In [*]:
In [*]:
Decision Tree Classification
In [*]:
In [*]:
In [*]:
In [*]:
Random Forest Classifier
In [*]:
In [*]:
In [*]:
pred_lr=LR.predict(xv_test)
LR.score(xv_test, y_test)
print(classification_report(y_test,pred_lr))
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)
print(classification_report(y_test, pred_dt))
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)
In [*]:
In [ ]:
In [ ]:
In [ ]:
In [ ]:
In [ ]:
print(classification_report(y_test, pred_rfc))[fake news prediction - Jupyter Notebook.pdf](https://github.com/Lochanitin/fake-news-detection/files/12193518/fake.news.prediction.-.Jupyter.Notebook.pdf)
