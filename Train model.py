#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.util import *


# ## This is NOT the raw data. It is each file summarized down into a single row if using the '5sec' model, or 5 rows if using the '1sec' model

# In[2]:


df_5sec = pd.read_csv(f'../data/complex_mafaulda_5sec.csv')
df_1sec = pd.read_csv(f'../data/complex_mafaulda_1sec.csv')
df_5sec.head()


# In[3]:


# make our class dictionary
keys = list(np.unique(df_5sec['class']))
codes = [int(_) for _ in np.arange(0,len(keys))]
classDict = dict(zip(keys,codes))

df_5sec['CLASS'] = df_5sec['class']
df_5sec['class'] = df_5sec['class'].map(classDict)

df_1sec['CLASS'] = df_1sec['class']
df_1sec['class'] = df_1sec['class'].map(classDict)

classDict


# # First, do train/test split for 5sec data
# # Then, use the same split for 1sec data
# # This way, any file can be used to compare accuracy between the two models

# In[4]:


drop = ['class','path','CLASS']
if 'startPoint' in df_5sec:
    drop.append('startPoint')
X5 = df_5sec.drop(columns=drop)
y5 = df_5sec['class']

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2, 
                                                    shuffle=True, random_state=45,
                                                    stratify=None)

print(f'Length of train set: {len(X5_train)}')
print(f'Length of test set: {len(X5_test)}')


# In[5]:


# used by the confusion_hist_plot function
target = 'class'

# define the model
model = RandomForestClassifier()

# train the model
model.fit(X5_train, y5_train)

# try to predict the data points we set aside for testing
preds5 = model.predict(X5_test)

#plot our results
confusion_hist_plot(df=df_5sec,y_test=y5_test,preds=preds5,codes=classDict);

# save the model
joblib.dump(model, f'./models/rfc_5sec.joblib')


# In[6]:


# These indices tell us which files we will need to pull from to re-summarize subsamples of the data for
# our sliding window
idxs = X5_test.index
idxs = [int(_) for _ in idxs]

val_files5 = df_5sec[['path','class','CLASS']].iloc[idxs]
val_files5.head()
val_files5.to_csv(f'./utils/test_files_5sec.csv')


# # For every file in the 5sec dataset, there are 5 files in the 1sec dataset. This is how to find those indices in the 1sec dataset

# In[7]:


index1 = []
for i in X5_train.index:
    I = i*5
    index1.extend([I,I+1,I+2,I+3,I+4])


# In[8]:


train1 = df_1sec.loc[index1]
if 'startPoint' in df_1sec:
    drop.append('startPoint')
X1_train = train1.drop(columns=drop)
y1_train = train1['class']

test1 = df_1sec.drop(index=index1)
X1_test = test1.drop(columns=drop)
y1_test = test1['class']


# In[9]:


# used by the confusion_hist_plot function
target = 'class'

# define the model
model = RandomForestClassifier()

# train the model
model.fit(X1_train, y1_train)

# try to predict the data points we set aside for testing
preds1 = model.predict(X1_test)

#plot our results
confusion_hist_plot(df=df_1sec,y_test=y1_test,preds=preds1,codes=classDict);
joblib.dump(model, f'./models/rfc_1sec.joblib')


# In[10]:


# These indices tell us which files we will need to pull from to re-summarize subsamples of the data for
# our sliding window
idxs = X1_test.index
idxs = [int(_) for _ in idxs]

val_files1 = df_1sec[['path','class','CLASS']].iloc[idxs]
val_files1.head()
val_files1.to_csv(f'./utils/test_files_1sec.csv')


# # Print out of some example files from each class. You will manually need to enter these file names into a config file for tests with specific file names

# In[14]:


for key in classDict.keys():
    print(key)
    print(val_files5['path'][val_files5['CLASS'] == key].head())
    print('---------------------------')


# In[ ]:




