import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import os
import plotly.express as px
from scipy.fft import fft, fftfreq
import statistics
from scipy import stats
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import *

from IPython.display import clear_output

import yaml
import json
import time
import joblib
import itertools

# This function will probably be of most use to you. Provide the path to a data file and the state of the machine
# captured in that data and return a nicely formatted DataFrame. For example, if you want the very first file of
# the dataset, you would say:
#
# df = new_csv(csv='/path/to/MAFAULDA/data/full/normal/12.288.csv', 'normal')
#
# If you wanted something other than 'normal', you would just say:
#
# df = new_csv(csv='/path/to/MAFAULDA/data/full/imbalance/6g/13.9264.csv', 'imbalance')
#
# for the imbalance, you could try and classify whether there is an added weight of 6g,10g,15g,etc.. but I have
# kept the classification broad so far
def new_csv(csv=None,fault=None):
    df = pd.read_csv(csv)
    missing = []
    [missing.append(float(d)) for d in df.columns]
    df.columns = ['tachometerSignal','axialUnderhang','radialUnderhang',
                  'tangentialUnderhang','axialOverhang','radialOverhang',
                  'tangentialOverhang','microphone']
    tempDf = pd.DataFrame(data=[missing],columns=df.columns,index=[0])
    df = pd.concat([tempDf,df],axis=0).reset_index().drop(columns=['index'])
    df['class'] = fault
    return df

# Check to see if a directory is a leaf or not. This is used later on for collecting data.
def is_leaf(path='.'):
    cwd = os.getcwd()
    os.chdir(path)
    check = sorted([f for f in os.listdir('./') if not f.startswith('.')], key=str.lower)
    os.chdir(cwd)
    if '.csv' in check[0]:
        return True
    else:
        return False
# Similar to reap_all_data() (found further down) but only collects vert simple data. Not used at the moment
def reap_stat_data(path='.', df=None, dictionary=None, columns=None):
    cwd = os.getcwd()
    os.chdir(path)
    files = sorted([f for f in os.listdir('./') if not f.startswith('.')], key=str.lower)
    for f in files:
        data = pd.read_csv(f)
        missing = []
        [missing.append(float(d)) for d in data.columns]
        data.columns = ['tachometerSignal','axialUnderhang','radialUnderhang',
                  'tangentialUnderhang','axialOverhang','radialOverhang',
                  'tangentialOverhang','microphone']
        missingDf = pd.DataFrame(data=[missing],columns=data.columns,index=[0])
        data = pd.concat([missingDf,data],axis=0)
        stats = data.describe()
        stats = stats.drop(index=['count','25%','50%','75%'],axis=0)
        tempDf = pd.DataFrame(columns=df.columns)
        for col in columns[:-1]:
            tempDf[col] = [stats[col[:-3]].iloc[dictionary[col[-3:]]]]
        classStr = path.split('/')
        tempDf['class'] = classStr[5]
        df = pd.concat([tempDf,df],axis=0)
    os.chdir(cwd)
    return df.reset_index().drop(columns=['index'])

# def walk_dirs(columns=None,codes=None,inputFrame=None,
#               finalFrame=None, walkDir=None):

#     for root, subdirs, files in os.walk(walkDir):

#         list_file_path = os.path.join(root, '')

#         print(list_file_path)
#         if is_leaf(path=list_file_path):
#             print('leaf')
#             data = reap_all_data(path=list_file_path,df=inputFrame,
#                                   dictionary=codes,columns=columns)
#             finalFrame = pd.concat([finalFrame,data],axis=0)

#         else:
#             print('not a leaf')
#     return finalFrame.reset_index().drop(columns='index')

# Given a parent directory, scan all sub-directories for data files and consolidate rows using SUMMARY STATISTICS
#
def walk_all_dirs(inputFrame=None,finalFrame=None, walkDir=None):

    for root, subdirs, files in os.walk(walkDir):

        list_file_path = os.path.join(root, '')

        print(list_file_path)
        if is_leaf(path=list_file_path):
            print('leaf')
            data = reap_all_data(path=list_file_path,df=inputFrame)
            finalFrame = pd.concat([finalFrame,data],axis=0)

        else:
            print('not a leaf')
    return finalFrame.reset_index().drop(columns='index')

def list_all_files(walkDir=None):
    files = []
    for root, subdirs, files in os.walk(walkDir):

        list_file_path = os.path.join(root, '')

        print(list_file_path)
        if is_leaf(path=list_file_path):
            files.append(sorted([f for f in os.listdir('./') if not f.startswith('.')], key=str.lower))
            print('leaf')
        else:
            print('not a leaf')
    return files

# Given a raw signal and our sampling rate, figure out the fundamental frequency of the signal using the method
# laid out in the papers.
def estimate_fundamental_frequency(sample=None,samp_per_sec=None):

    # values of the tachometer signal
    y = np.array(sample)
    # sample rate
    T = 1/samp_per_sec

    # calculate the Discrete Fourier Transform of the tachometer signal and
    # return the x values (frequency (Hz)) and the y values (magnitude)
    xf, yf = do_fft(y=y,T=T)

#     print(f'FROM EFF -- len: {len(xf)}')

    # initialize empty list that will hold candidates for the fundamental
    # frequency
    candidates = []
    idxs = []

    # scan the DFT for frequencies that may be candidates for the fundamental frequency.
    # Once a candidate has been found, mask it and the surrouding area to hone in on another.
    # return the list of candidates along with the modified x and y values for the DFT.
    for i in range(4):
        # find the frequency with the highest magnitude and return the x and y values
        maxIdx, argmax = find_highest_frequency(xf,yf)
#         print(f'FROM EFF -- maxIdx: {maxIdx}')
        # store the value of this frequency in candidates
        candidates.append(argmax)
        idxs.append(maxIdx)
        # mask the located frequency and it's neighbors [-3,+3] with zeros
#         zone = yf[maxIdx-3:maxIdx+4]
#         print(zone.shape)
#         if zone.shape != (7,):
#             print(yf)
#         zone = np.zeros(7)

        lo = maxIdx-3
        hi = maxIdx+4

        # this covers the case where the argmax is {0,1,2} and would cause the initial index to be less than 0
        if lo < 0:
            lo-=lo

        # if everything is working as usual, width should be 7
        # else, width becomes a smaller value. This means we are zeroing out an area from [0,hi+1]
        width = hi-lo

        yf[lo:hi] = np.zeros(width)


#         yf[maxIdx-3:maxIdx+4] = np.zeros(7)

    return candidates, idxs, xf, yf

# Given the X and Y spectral coordinates, find the highest peak and return the magnitude and the frequency
def find_highest_frequency(xf=None,yf=None):

    maxIdx = np.argmax(yf)
    argmax = xf[maxIdx]

    return maxIdx, argmax

# Given a raw signal, for example the Tachometer Signal from full/normal/12.288.csv, and 1/(sampling frequency
# in Hz), return the X and Y coordinates of the spectrum. In our case, the sampling frequency is 50kHz, or 50000Hz
def do_fft(y=None,T=None):


    N = len(y)

    yf = fft(y)[0:N//2]
    xf = fftfreq(N,T)[:N//2]
    yf = 2.0/N * np.abs(yf)

    return xf, yf

# Given the X and Y coordinates gathered from the above function do_fft(), simply plot them out in an interactive
# manner using plotly
def plot_fft(xf=None,yf=None):

    f = pd.DataFrame(columns={'x','y'})
    f['x'] = xf
    f['y'] = yf
    fig = px.line(f,x='x',y='y', title='Spectral Analysis (FFT)',
                  labels=dict(
                         x='frequency',
                         y='magnitude')
                   )

    return fig

# Given a parent directory, scan for data files and consolidate all 250,000 rows in each file
# down to 1 using the supplied functions. This converts raw data into SUMMARY STATISTICS and should only be used
# when not wanting to work with the entire dataset. The data gathered from this might be useful just to get
# something running as a proof of concept, but it does take some time to consilidate all the data. Called by
# walk_all_dirs()
def reap_all_data(path='.', df=None):
    columns = df.columns
    cwd = os.getcwd()
    os.chdir(path)
    files = sorted([f for f in os.listdir('./') if not f.startswith('.')], key=str.lower)
    for f in files:
        data = pd.read_csv(f)
        missing = []
        [missing.append(float(d)) for d in data.columns]
        data.columns = ['tachometerSignal','axialUnderhang','radialUnderhang',
                  'tangentialUnderhang','axialOverhang','radialOverhang',
                  'tangentialOverhang','microphone']
        missingDf = pd.DataFrame(data=[missing],columns=data.columns,index=[0])
        data = pd.concat([missingDf,data],axis=0)
        tempDf = pd.DataFrame(columns=df.columns)
        candidates, idxs, xf, yf= estimate_fundamental_frequency(sample=data['tachometerSignal'],
                                                                 samp_per_sec=50000)
        #print(np.min(candidates))

        #print(columns)
        for col in columns[:-2]:

            _col = col.split('-')
            #print(col)
            featureName = _col[0] # Ex: axialOverhang
            statName = _col[1]    # Ex: Mean
            #print(f'featureName: {featureName}')
            #print(f'statName: {statName}')

            if statName == 'Entropy':
                tempDf[col] = [stats.entropy(data[featureName].value_counts())]
                #break
            elif statName == 'Kurtosis':
                tempDf[col] = [stats.kurtosis(data[featureName])]
                #break
            elif statName == 'Mean':
                tempDf[col] = [np.mean(data[featureName])]
                #break
            ##############################################
            elif statName == 'Std':
                tempDf[col] = [np.std(data[featureName])]
                #break
            elif statName == 'Min':
                tempDf[col] = [np.min(data[featureName])]
                #break
            elif statName == 'Max':
                tempDf[col] = [np.max(data[featureName])]
                #break
            ##############################################
            elif statName == 'F':
                spectrals = extract_spectral_feature(sample=np.array(data[featureName]),
                                                     samp_per_sec=50000,
                                                     idxs=idxs)
                tempDf[col] = [spectrals[0][0]]
                #break
            elif statName == '2F':
                spectrals = extract_spectral_feature(sample=np.array(data[featureName]),
                                                     samp_per_sec=50000,
                                                     idxs=idxs)
                tempDf[col] = [spectrals[0][1]]
                #break
            elif statName == '3F':
                spectrals = extract_spectral_feature(sample=np.array(data[featureName]),
                                                     samp_per_sec=50000,
                                                     idxs=idxs)
                tempDf[col] = [spectrals[0][2]]
                #break
        classStr = path.split('/')
        tempDf['fundamentalFrequency'] = np.min(candidates)
        tempDf['path'] = os.getcwd().split('data/')[-1]+'/'+f
        tempDf['class'] = os.getcwd().split('full/')[-1]
        #print(tempDf.head())
        df = pd.concat([tempDf,df],axis=0)
    os.chdir(cwd)
    return df.reset_index().drop(columns=['index'])

# extract the spectral features laid out in the paper.
# given a sample, or in more exact terms, all of the rows corresponding to a single column in a data file,
# AKA a whole signal, take the FFT and find the magnitude of the spectrum at F, 2F, and 3F. F is the fundamental
# frequency of a signal and can be found by estimate_fundamental_frequency()
def extract_spectral_feature(sample=None,samp_per_sec=None,idxs=None):

    # do fft and return x and y coordinates
    xf,yf = do_fft(y=sample,T=1/samp_per_sec)

    # these are the candidates for the fundamental frequency and just so happen to be F, 2F, 3F, and 4F
    idxs.sort()

    # we only need F, 2F, and 3F
    spectrals = [yf[idxs[0]],
                 yf[idxs[1]],
                 yf[idxs[2]]]

    return spectrals, xf,yf

# not used. can find 2*F and 3*F with the candidates list
def find_nearest_peak(idx=None,yf=None):

    right = idx + 1
    left = idx - 1
    prev = idx

    if yf[right] > yf[left]:
        current = right
        while yf[current] > yf[prev]:
            current+=1
            prev+=1

    elif yf[left] > yf[right]:
        current = left
        while yf[current] > yf[prev]:
            current-=1
            prev-=1

    return prev

def list_all_files(walkDir=None):
    f = []
    for root, subdirs, files in os.walk(walkDir):

        list_file_path = os.path.join(root, '')

#         print(list_file_path)
        if is_leaf(path=list_file_path):
            t = sorted([f for f in os.listdir(list_file_path) if not f.startswith('.')], key=str.lower)
            [f.append(list_file_path+_) for _ in t]
#             f.append(t)
#             print(f'{list_file_path} -> leaf')
#         else:
#             print(f'{list_file_path} -> not a leaf')
    return f

# Plots a confusion matrix next to a histogram to show the ratio of TF/FP TN/FN after predicting with our
# classifier model. The purpose of the histogram is to show the distribution of each of the classes since there
# is some notable disparity in this dataset. Ex: normal only has 49 measurements wheres imbalance has 333
# measurements (data files)
# codes: classes get ingested by the classifier model as numerical values, so codes is just a dictionary that
# says which value corresponds to which class
def confusion_hist_plot(df=None,y_test=None,preds=None,codes=None):

    target = 'class'

    fig, ((ax1),(ax2)) = plt.subplots(1,2,figsize=(20,8),sharey=False)

    cm = confusion_matrix(y_test,preds,normalize='true')
    _cm = confusion_matrix(y_test,preds,normalize=None)
    sns.heatmap(cm, annot=_cm, cmap='Blues',ax=ax1)

    v = [*y_test.values]
    v.extend([*preds])
    v = [*set(v)]
    v = pd.Series(v).map(dict(map(reversed,codes.items())))
    keys_to_extract = v.values
    codes = {key: codes[key] for key in keys_to_extract}

    ax1.set_xticklabels(codes.keys(),rotation=45)
    ax1.set_yticklabels(codes.keys(),rotation=0)

    histData = pd.DataFrame(data=y_test,columns=[target])

    keyDf=histData[target].map(dict(map(reversed,codes.items())))
    orderedDf = pd.Categorical(keyDf,categories=[*codes.keys()],ordered=True)
    sns.histplot(y=orderedDf,ax=ax2,color='cornflowerblue',linewidth=1.75)
#     ax2.set_yticklabels(classDict.keys(),rotation=45)

    acc = accuracy_score(y_test,preds)
    plt.suptitle(f'Confusion Matrix With Supporting Histogram\naccuracy score: {acc:.4f}')
    return fig
#     fig.suptitle("Confusion Matrix With Supporting Histogram", fontsize=14)

def summarize(df=None,columns=None):
    columns = columns
    data = df.copy()
    tempDf = pd.DataFrame()
    classes = np.unique(data['class'])
#     print(classes)
#     print(data.index)

    if len(classes) > 1:
        tempDf['class'] = ['mixed']
    else:
        tempDf['class'] = [classes[0]]

    data = data.drop(columns=['class'])

    candidates, idxs, xf, yf= estimate_fundamental_frequency(sample=data['tachometerSignal'],
                                                             samp_per_sec=50000)
    #print(np.min(candidates))

    #print(columns)
    for col in columns[:-2]:

        _col = col.split('-')
        #print(col)
        featureName = _col[0] # Ex: axialOverhang
        statName = _col[1]    # Ex: Mean
        #print(f'featureName: {featureName}')
        #print(f'statName: {statName}')

        if statName == 'Entropy':
            tempDf[col] = [stats.entropy(data[featureName].value_counts())]
            #break
        elif statName == 'Kurtosis':
            tempDf[col] = [stats.kurtosis(data[featureName])]
            #break
        elif statName == 'Mean':
            tempDf[col] = [np.mean(data[featureName])]
            #break
        ##############################################
        elif statName == 'Std':
            tempDf[col] = [np.std(data[featureName])]
            #break
        elif statName == 'Min':
            tempDf[col] = [np.min(data[featureName])]
            #break
        elif statName == 'Max':
            tempDf[col] = [np.max(data[featureName])]
            #break
        ##############################################
        elif statName == 'F':
            spectrals = extract_spectral_feature(sample=np.array(data[featureName]),
                                                 samp_per_sec=50000,
                                                 idxs=idxs)
            tempDf[col] = [spectrals[0][0]]
            #break
        elif statName == '2F':
            spectrals = extract_spectral_feature(sample=np.array(data[featureName]),
                                                 samp_per_sec=50000,
                                                 idxs=idxs)
            tempDf[col] = [spectrals[0][1]]
            #break
        elif statName == '3F':
            spectrals = extract_spectral_feature(sample=np.array(data[featureName]),
                                                 samp_per_sec=50000,
                                                 idxs=idxs)
            tempDf[col] = [spectrals[0][2]]
            #break
#     classStr = path.split('/')
    tempDf['fundamentalFrequency'] = np.min(candidates)
#     path = np.unique(df['path'])
#     tempDf['path'] = path
#     tempDf['class'] = [p.split('full/')[-1] for p in path]
    #print(tempDf.head())
    return tempDf.reset_index().drop(columns=['index'])

def plot_window( df=None, n=2, back_colors=['r','b'],
                 trues=None, preds=None,
                 fig_size=(14,26), p_color='C0', t_color='C1',
                 p_alpha=0.8, t_alpha=0.4, class_dict=None,
                 window_width=10000, width_per_step=10000):

    data_reindexed = df.reset_index().drop(columns=['index'])
    l = len(data_reindexed)/n
    sigs = data_reindexed.columns[:-2]
    n_plots = len(sigs)+1

    fig = plt.figure()

    # to change size of subplot's
    # set height of each subplot as 8
    fig.set_figheight(16)

    # set width of each subplot as 8
    fig.set_figwidth(14)

    # create grid for different subplots
    spec = gridspec.GridSpec(ncols=1, nrows=n_plots,
                             width_ratios=[1], wspace=0.5,
                             hspace=0.5, height_ratios= [*np.ones(n_plots-1)]+[2])

    for i in range(n_plots):
        tick = len(data_reindexed)/5
        xticks = np.arange(0,len(data_reindexed)+tick,tick)
        ax = fig.add_subplot(spec[i])
        if i < n_plots-1:
            ax.plot(data_reindexed[sigs[i]],label=sigs[i])
            for _ in range(n):
                ax.axvspan(l*_, (l*_)+l, facecolor=back_colors[_%2], alpha=0.13)
            ax.axvspan(50000,50000+window_width,facecolor='y',alpha=.35)
            ax.set_xticks(xticks)
            ax.set_xticklabels(['']*len(xticks))
        else:
#             ax.scatter(np.arange(window_width,(l*n)+width_per_step,width_per_step),preds,
#                        label='predicted class',facecolors='none',edgecolors=p_color,alpha=p_alpha)
#             ax.scatter(np.arange(window_width,(l*n)+width_per_step,width_per_step),trues,
#                        label='true class',facecolors='none',edgecolors=t_color,alpha=t_alpha)
            ax.scatter(np.arange(window_width,(l*n),width_per_step),preds,
                       label='predicted class',facecolors='none',edgecolors=p_color,alpha=p_alpha)
            ax.scatter(np.arange(window_width,(l*n),width_per_step),trues,
                       label='true class',facecolors='none',edgecolors=t_color,alpha=t_alpha)
            ax.set_yticks([*class_dict.values()])
            ax.set_yticklabels([*class_dict.keys()])
            ax.set_xticks(xticks)
            for _ in range(n):
                ax.axvspan(l*_, (l*_)+l, facecolor=back_colors[_%2], alpha=0.13)
        ax.legend()
    acc = accuracy_score(trues[trues!=6],pd.Series(preds)[trues!=6])
    fig.suptitle(f'\n\n\n\n{window_width} wide window and {width_per_step} width per step\nAccuracy: {round(acc*100,2)}%')
#     plt.show()
    return fig, acc

#     fig, axs = plt.subplots(n_plots,figsize=fig_size,sharex=True)

#     for i,ax in enumerate(axs):
#         if i < n_plots-1:
#             ax.plot(data_reindexed[sigs[i]],label=sigs[i])
#             for _ in range(n):
#                 ax.axvspan(l*_, (l*_)+l, facecolor=back_colors[_%2], alpha=0.13)
#             ax.axvspan(50000,10000,facecolor='y',alpha=.35)
#         else:
#             ax.scatter(np.arange(window_width,(l*n)+width_per_step,width_per_step),preds,
#                        label='predicted class',facecolors='none',edgecolors=p_color,alpha=p_alpha)
#             ax.scatter(np.arange(window_width,(l*n)+width_per_step,width_per_step),trues,
#                        label='true class',facecolors='none',edgecolors=t_color,alpha=t_alpha)
#             ax.set_yticks([*class_dict.values()])
#             ax.set_yticklabels([*class_dict.keys()])
#             for _ in range(n):
#                 ax.axvspan(l*_, (l*_)+l, facecolor=back_colors[_%2], alpha=0.13)
#         ax.legend()
    fig.suptitle(f'{window_width} window width and {width_per_step} width per step')

def prepare_window_data( data_path='~/ML/MAFAULDA/data/', steps_per_file=250000, num_files=2,
                         sensors=None, file_idxs=None ):
    df = pd.DataFrame(columns=sensors)
    val_files = pd.read_csv('/Users/nrprzybyl/ML/MAFAULDA/window/utils/val_files.csv').set_index('Unnamed: 0')
    if file_idxs is None:
        for i in range(num_files):
            p = data_path+val_files['path'].iloc[i]
            tempDf = pd.read_csv(p,header=None)
            tempDf.columns = sensors
            tempDf['path'] = [p]*steps_per_file
            tempDf['class'] = [p.split('full/')[-1].split('/')[0]]*steps_per_file
            df = pd.concat([df,tempDf])
    else:
        num_files = len(file_idxs)
        for i in file_idxs:
            p = data_path+val_files[val_files.index == i]['path'].values[0]
            tempDf = pd.read_csv(p,header=None)
            tempDf.columns = sensors
            tempDf['path'] = [p]*steps_per_file
            tempDf['class'] = [p.split('full/')[-1].split('/')[0]]*steps_per_file
            df = pd.concat([df,tempDf])

    return df

def slide_window( window_width=10000,width_per_step=10000,window_model=None,
                  num_files=2, df=None, columns=None, steps_per_file=250000,
                  classDict=None ):

    # sliding window hyper-parameters
#     window_width = 10000
#     width_per_step = 2500
#     window_model = model

    preds, trues = [],[]
#     for i in range(0,(steps_per_file*num_files)-window_width+width_per_step,width_per_step):
    for i in range(0,(steps_per_file*num_files)-window_width,width_per_step):
        data = df.iloc[i:i+window_width]
#         print(f'{i}:{i+window_width}')
        summarized = summarize(df=data,columns=columns)
        X = summarized.drop(columns=['class'])
        y = summarized['class'].map(classDict).values[0]
        pred = window_model.predict(X)[0]
        preds.append(pred)
        trues.append(y)

    # unique_values = []
    # unique_values.extend(preds)
    # unique_values.extend(trues)
    # unique_values = [*set(unique_values)]

    trues = pd.Series(trues)
    return trues, preds

def plot_feature_importance(model=None, X=None, figsize=None):
    feature_names = X.columns
    importance_frame = pd.DataFrame()
    importance_frame['Features'] = X.columns
    importance_frame['Importance'] = model.feature_importances_ #coef_
    importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)

    plt.figure(figsize=figsize)
    plt.barh(np.arange(1,len(X.columns)+1), importance_frame['Importance'], align='center', alpha=0.5)
    plt.yticks(np.arange(1,len(X.columns)+1), importance_frame['Features'])
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', titlesize=16)
    return importance_frame

def sweep_window(config='/Users/nrprzybyl/ML/MAFAULDA/window/config/config.yaml'):
    t = time.time()
    with open(config,'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with open('/Users/nrprzybyl/ML/MAFAULDA/window/utils/utils.json', 'r') as file:
        utils = json.load(file)

    # load the model
    model = joblib.load('./models/rfc.joblib')

    idxs = utils['idxs']
    S = utils['signals']
    columns = utils['columns']
    classDict = utils['classes']

    default = conf['default']
    plot_path = default['plot_path']
    experiment_name = default['experiment_name']
    path = os.path.join(plot_path,experiment_name)
    isDir = os.path.isdir(path)
    if not isDir:
        os.mkdir(path)

    sweep = conf['sweep']
    random_pick = sweep['random_pick']
    if not random_pick:
        file_idxs = sweep['file_idxs']
        n_files = len(file_idxs)
    else:
        n_files = sweep['n_files']
        file_idxs = [*np.random.choice([*idxs],n_files,replace=False)]
    width_per_step = sweep['width_per_step']
    window_width = sweep['window_width']
    wps = np.arange(width_per_step['lo'],width_per_step['hi']+width_per_step['step'],width_per_step['step'])
    ww = np.arange(window_width['lo'],window_width['hi']+window_width['step'],window_width['step'])
    params = [*itertools.product(wps,ww)]
    n_iters = len(params)

    meta = {}

    for i,p in enumerate(params):
        df = prepare_window_data(sensors=S, file_idxs=file_idxs)

        _width_per_step = p[0]
        _window_width = p[1]

        if i % 1 == 0:
            clear_output(wait=True)
            print(f'{i+1}/{n_iters} -- {round(100*((i+1)/(n_iters)),2)}%')

        _t = time.time()
        trues, preds = slide_window(window_model=model, df=df, columns=columns,
                                    num_files=n_files, width_per_step=_width_per_step,
                                    window_width=_window_width,classDict=classDict)
        _t = time.time() - _t

        overall_acc = accuracy_score(trues,preds)

        ######################################################

        t_zones = [ [] for _ in np.ones(n_files) ]
        p_zones = [ [] for _ in np.ones(n_files) ]

        zone_idxs = [ int((i*wps) // 250000) for i in range(len(trues)) ]

#         print(zone_idxs)
        for i,z in enumerate(zone_idxs):
#             print(z)
            p_zones[z].append(preds[i])
            t_zones[z].append(trues[i])

#         print(t_zones)
#         print(p_zones)

        ######################################################

        zones = []

        fig, acc = plot_window(df,trues=trues,preds=preds,class_dict=classDict, n=n_files, width_per_step=_width_per_step, window_width=_window_width)
        fig_name = ''
        if sweep['random_pick'] is True:
            for _ in file_idxs:
                fig_name+=f'{_}_'
        else:
            s1,s2 = '',''
            for i in range(5):
                s1+=f'{file_idxs[i]}_'
                s2 = f'{file_idxs[-i-1]}_'+s2
            fig_name = f'{s1[:-1]}\...{s2[:-1]}'    # 0_1_2_3_4..._n-5,n-4,n-3,n-2,n-1
            # fig_name+=f'{*file_idxs[:5]}\...{*file_idxs[-5:]}_'
        p = os.path.join(path,f'{_window_width}_{_width_per_step}')
        isDir = os.path.isdir(p)
        if not isDir:
            os.mkdir(p)
        run = f'{_window_width}_{_width_per_step}'
        fig_path = f'{path}/{run}/{fig_name[:-1]}.png'
        fig.savefig(fig_path)
        plt.close()

        fig1 = confusion_hist_plot(df=df,y_test=trues,preds=preds,codes=classDict)
        fig1.savefig(f'{fig_path} confusion.png')
        plt.close()

        meta[run] = {}
        meta[run]['overall_acc'] = overall_acc
        meta[run]['acc'] = acc

        meta[run]['time'] = _t

        ######################################################

        meta[run]['zones'] = {}
        meta[run]['zones'] = [ {} for _ in np.ones(n_files) ]
        for i in range(n_files):
            meta[run]['zones'][i]['acc'] = accuracy_score(t_zones[i],p_zones[i])

    file_paths = [*df['path'].unique()]
    meta[run]['file_paths'] = file_paths

        ######################################################

    t = time.time() - t
    meta[run]['time'] = t
    with open(f'{path}/out', 'w') as file:
        file.write(json.dumps(meta))
