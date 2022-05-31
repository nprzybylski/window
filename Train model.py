#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.util import *


# ## Change this string to '5sec' if you want to train a new model that uses all 5 seconds of raw data
# ## Change this string to '1sec' if you want to train a new model that uses 1 second slices of raw data

# In[2]:


train_width = '1sec'


# ## This is NOT the raw data. It is each file summarized down into a single row if using the '5sec' model, or 5 rows if using the '1sec' model

# In[3]:


df = pd.read_csv(f'../data/complex_mafaulda_{train_width}.csv')
df.head()


# In[4]:


# make our class dictionary
keys = list(np.unique(df['class']))
codes = [int(_) for _ in np.arange(0,len(keys))]
classDict = dict(zip(keys,codes))
classDict['mixed'] = 6

df['CLASS'] = df['class']
df['class'] = df['class'].map(classDict)

classDict


# In[5]:


drop = ['class','path','CLASS']
if 'startPoint' in df:
    drop.append('startPoint')
X = df.drop(columns=drop)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    shuffle=True, random_state=45,
                                                    stratify=None)

print(f'Length of train set: {len(X_train)}')
print(f'Length of test set: {len(X_test)}')


# In[6]:


# used by the confusion_hist_plot function
target = 'class'

# define the model
model = RandomForestClassifier()

# train the model
model.fit(X_train, y_train)

# try to predict the data points we set aside for testing
preds = model.predict(X_test)

#plot our results
confusion_hist_plot(df=df,y_test=y_test,preds=preds,codes=classDict);

# save the model
joblib.dump(model, f'./models/rfc_{train_width}.joblib')


# In[7]:


# These indices tell us which files we will need to pull from to re-summarize subsamples of the data for
# our sliding window
idxs = X_test.index
idxs = [int(_) for _ in idxs]

val_files = df[['path','class','CLASS']].iloc[idxs]
val_files.head()
val_files.to_csv(f'./utils/test_files_{train_width}.csv')


# In[ ]:




