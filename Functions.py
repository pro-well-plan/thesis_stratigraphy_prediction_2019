# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: conda_tensorflow_p36
#     language: python
#     name: conda_tensorflow_p36
# ---

# +
import os
import sys
data_dir = "/home/ec2-user/pwp-summer-2019/master_thesis_nhh_2019/processed_data/" 
raw_dir = "/home/ec2-user/pwp-summer-2019/master_thesis_nhh_2019/raw_data/" 

import pandas as pd
import numpy as np
import random
import math

pd.set_option('display.max_columns', 999)


# -

# ### Function for splitting the data sets based on formation-distribution

# +
# Inspired by: https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit

def StratifiedGroupShuffleSplit(
    df_main,
    train_proportion=0.6,
    val_proportion = 0.3,
    hparam_mse_wgt = 0.1,
    df_group="title",
    y_var="formation_2",
    norm_keys=['gr','tvd','rdep'],
    seed = 42
):
    np.random.seed(seed) # Set seed
    df_main.index = range(len(df_main)) # Create unique index for each observation in order to reindex
    df_main = df_main.reindex(np.random.permutation(df_main.index)) # Shuffle dataset

    # Create empty train, val and test datasets
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()

    hparam_mse_wgt = hparam_mse_wgt # Must be between 0 and 1
    assert(0 <= hparam_mse_wgt <= 1)
    train_proportion = train_proportion # Must be between 0 and 1
    assert(0 <= train_proportion <= 1)
    val_proportion = val_proportion # Must be between 0 and 1
    assert(0 <= val_proportion <= 1)
    test_proportion = 1-train_proportion-val_proportion # Remaining in test proportion
    assert(0 <= test_proportion <= 1)
    
    # Group the data set
    subject_grouped_df_main = df_main.groupby([df_group], sort=False, as_index=False) 
    # Find the proportion of the total for each category
    category_grouped_df_main = df_main.groupby(y_var).count()[[df_group]]/len(df_main)*100 

    # Functoin for calculating MSE
    def calc_mse_loss(df):
        # Find the proportion of the total for each category in the specific data set
        grouped_df = df.groupby(y_var).count()[[df_group]]/len(df)*100
        # Merge the data set above with the original proportion for each category
        df_temp = category_grouped_df_main.join(grouped_df, on = y_var, how = 'left', lsuffix = '_main')
        # Fill NA
        df_temp.fillna(0, inplace=True)
        # Square the difference
        df_temp['diff'] = (df_temp[df_group+'_main'] - df_temp[df_group])**2
        # Mean of the squared difference
        mse_loss = np.mean(df_temp['diff'])
        return mse_loss

    # Initialize the train/val/test set
    # First three wells are assigned to train/val/test
    i = 0
    for well, group in subject_grouped_df_main:
        group = group.sort_index()
        if (i < 3):
            if (i == 0):
                df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
            elif (i == 1):
                df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
            else:
                df_test = df_test.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
        
        # Caluclate the difference between previous dataset and the one in the loop
        mse_loss_diff_train = calc_mse_loss(df_train) - calc_mse_loss(df_train.append(pd.DataFrame(group), 
                                                                                      ignore_index=True))
        mse_loss_diff_val = calc_mse_loss(df_val) - calc_mse_loss(df_val.append(pd.DataFrame(group), 
                                                                                ignore_index=True))
        mse_loss_diff_test = calc_mse_loss(df_test) - calc_mse_loss(df_test.append(pd.DataFrame(group), 
                                                                                   ignore_index=True))
        
        # Calculate the total lenght so far
        total_records = df_train.title.nunique() + df_val.title.nunique() + df_test.title.nunique()

        # Calculate how far much much is left before the goal is reached
        len_diff_train = (train_proportion - (df_train.title.nunique()/total_records))
        len_diff_val = (val_proportion - (df_val.title.nunique()/total_records))
        len_diff_test = (test_proportion - (df_test.title.nunique()/total_records))

        len_loss_diff_train = len_diff_train * abs(len_diff_train)
        len_loss_diff_val = len_diff_val * abs(len_diff_val)
        len_loss_diff_test = len_diff_test * abs(len_diff_test)

        loss_train = (hparam_mse_wgt * mse_loss_diff_train) + ((1-hparam_mse_wgt) * len_loss_diff_train)
        loss_val = (hparam_mse_wgt * mse_loss_diff_val) + ((1-hparam_mse_wgt) * len_loss_diff_val)
        loss_test = (hparam_mse_wgt * mse_loss_diff_test) + ((1-hparam_mse_wgt) * len_loss_diff_test)

        # Assign to either train, val or test
        if (max(loss_train,loss_val,loss_test) == loss_train):
            df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
        elif (max(loss_train,loss_val,loss_test) == loss_val):
            df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
        else:
            df_test = df_test.append(pd.DataFrame(group), ignore_index=True)

        i += 1
    
    return df_train, df_val, df_test
# -

# ### Function for setting up the LSTM data set

# +
# Inspired by: 
# https://github.com/blasscoc/LinkedInArticles/blob/master/WellFaciesLSTM/LSTM%20Facies%20Competition.ipynb
from sklearn.preprocessing import OneHotEncoder

def chunk(x, y, num_chunks, size=61, random=True):
    rng = x.shape[0] - size
    if random:
        indx = np.int_(
            np.random.rand(num_chunks) * rng) + size//2
    else:
        indx = np.arange(0,rng,1) + size//2
    Xwords = np.array([[x[i-size//2:i+size//2+1,:] 
                                for i in indx]])
    ylabel = np.array([y[i] for i in indx])
    
    return Xwords[0,...], ylabel

def _num_pad(size, batch_size):
    return (batch_size - np.mod(size, batch_size))

def setup_lstm_stratify(df,
               df_group='title',
               batch_size=128,
               wvars=['gr','tvd','rdep'],
               y_var = 'formation',
               win=9,
               n_val=39
              ):

    df = df.fillna(0)
    
    df_grouped = df.groupby([df_group], sort=False, as_index=False) 
    df_x = []
    df_y = []

    for key,val in df_grouped:
        val = val.copy() 
            
        _x = val[wvars].values
        _y = val[y_var].values
        
        __x, __y = chunk(_x, _y, 400, size=win, random=False)
        
        df_x.extend(__x)
        df_y.extend(__y)

    df_x = np.array(df_x)
    df_y = np.array(df_y)    
    
    #One Hot Encoding
    enc = OneHotEncoder(sparse=False, categories=[range(n_val)]) 
    df_y = enc.fit_transform(np.atleast_2d(df_y).T)
    df_x = df_x.transpose(0,2,1)

    # pad to batch size    
    num_pad = _num_pad(df_x.shape[0], batch_size)
    df_x = np.pad(df_x, ((0,num_pad),(0,0),(0,0)), mode='edge')
    df_y = np.pad(df_y, ((0,num_pad), (0,0)), mode='edge')
        
    return df_x, df_y


# -

# ### Data generator for feeding the LSTM model

# +
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_x, df_y, batch_size=128):
        'Initialization'
        self.df_x = df_x
        self.df_y = df_y
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.df_x))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df_x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index_epoch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        return (self.df_x[index_epoch], self.df_y[index_epoch])


# -

# ### Class for feature engineering and cleaning the data sets

class feature_engineering:
    def __init__(self,df, above_below_variables, num_shifts, cols_to_remove,thresh,
                 log_variables, y_variable, outlier_values, var1_ratio = 'gr'
                ):
        #Variables:
        self.original_df = df
        self.df = df
        self.above_below_variables = above_below_variables
        self.y_variable = y_variable
        self.num_shifts = num_shifts
        self.cols_to_remove = cols_to_remove
        self.thresh = thresh
        self.log_variables = log_variables
        self.var1_ratio = var1_ratio
        self.outlier_values = outlier_values
        self.var2_ratio = log_variables
        
    def log_values(self):
        'Calculates both the log values'
        for variable in self.log_variables:
            self.df[variable] = np.log(self.df[variable])      
        #self.df = self.df.drop(self.log_variables,axis = 1)
                      
    def above_below(self):
        'Add the value above and below for each column in variables'
        for var in self.above_below_variables:
            self.df[var+'_above'] = self.df.groupby('title')[var].shift(self.num_shifts)
            self.df[var+'_below'] = self.df.groupby('title')[var].shift(self.num_shifts)
        
        drop = [self.above_below_variables[0]+'_above', self.above_below_variables[0]+'_below']
        for i in drop:
            self.df = self.df.dropna(subset=[i])
            
    def var_ratio(self):
        'Generate the ratio of GR divided by specified variables'
        for var in self.var2_ratio:
            self.df[self.var1_ratio + '_' + var] = (self.df[self.var1_ratio]/self.df[var])
            self.df[self.var1_ratio + '_' + var].loc[self.df[self.var1_ratio + '_' + var] == float('Inf')] = 0
            self.df[self.var1_ratio + '_' + var].loc[self.df[self.var1_ratio + '_' + var] == -float('Inf')] = 0
        
    def cleaning(self):
        'Remove certain formations, rows with a lot of NAs and make y_variabl categorical'
        self.df = self.df.drop(self.cols_to_remove,axis = 1)
        self.df = self.df.dropna(thresh=self.thresh) #thresh= 12

        self.df = self.df[np.isfinite(self.df['tvd'])]
        self.df = self.df[(self.df.formation != 'water depth')]
    
        self.df[self.y_variable] = self.df[self.y_variable].astype('category')
        self.df = self.df[self.df[self.y_variable].cat.codes != -1]
        self.df.reset_index(inplace=True, drop = True)
        
    def xyz(self):
        'Lat/Long for ML purposes'
        self.df['x'] = np.cos(self.df['lat']) * np.cos(self.df['long'])
        self.df['y'] = np.cos(self.df['lat']) * np.sin(self.df['long'])
        self.df['z'] = np.sin(self.df['lat'])
        self.df = self.df.drop(['lat','long'],axis = 1)
        
    def single_pt_haversine(self,degrees=True):
        """
        'Single-point' Haversine: Calculates the great circle distance
        between a point on Earth and the (0, 0) lat-long coordinate
        """
        r = 6371 # Earth's radius (km). Have r = 3956 if you want miles

        # Convert decimal degrees to radians
        if degrees:
            lat, lng = map(math.radians, [self.df.lat, self.df.lng])

        # 'Single-point' Haversine formula
        a = math.sin(lat/2)**2 + math.cos(lat) * math.sin(lng/2)**2
        d = 2 * r * math.asin(math.sqrt(a)) 
        
        self.df['well_distance'] = [self.single_pt_haversine(x, y) for x, y in zip(lat, long)]
        
    def drop_new_values(self):  
        'NAs are introduced when we calculate above and below. This function removes them'
        drop = ["gr_above", "gr_below"]
        for i in drop:
            self.df = self.df.dropna(subset=[i])
        self.df = self.df
        
    def remove_outliers(self):
        for key,value in self.outlier_values.items():
            self.df = self.df[self.df[key] <= value]
            self.df = self.df[self.df[key] >= 0]
          
    def done(self):
        'Return the self.df set and a dictionary of formations and their corresponding number'
        self.remove_outliers()
        self.remove_outliers()
        self.log_values()
        self.above_below()
        self.cleaning()
        self.xyz()
        return self.df


# ### Visualizations

# +
formation_colors = ['#d96c6c', '#ffe680', '#336633','#4d5766', '#cc99c9', 
                 '#733939', '#f2eeb6', '#739978', '#333366', '#cc669c', 
                 '#f2b6b6', '#8a8c69', '#66ccb8', '#bfbfff', '#733950', 
                 '#b27159', '#c3d96c', '#336663', '#69698c', '#33262b', 
                 '#bfa38f', '#2d3326', '#1a3133', '#8f66cc', '#99737d', 
                 '#736256', '#65b359', '#73cfe6', '#673973', '#f2ba79', 
                 '#bef2b6', '#86aab3', '#554359', '#8c6c46', '#465943', 
                 '#73b0e6', '#ff80f6', '#4c3b26']
group_colors = ['#ff4400', '#cc804e', '#e5b800', '#403300', '#4da63f', 
                '#133328', '#00cad9', '#005fb3', '#0000f2', '#292259', 
                '#d052d9', '#33131c', '#ff6176']

def plot_well_comparison(df, well_index, formation_colors, group_colors, model_name = None, save = False):
    
    #df['group_2'] = df['group'].astype('category').cat.codes
    
    logs = df.loc[df["title"] == df.title.unique()[well_index]]
    #logs = logs.sort_values(by='tvd')
    
    cluster_predicted_formation=np.repeat(np.expand_dims(logs['predicted'].values,1), 100, 1)
    
    cluster_actual_formation=np.repeat(np.expand_dims(logs['formation_2'].values,1), 100, 1)
    
    cluster_predicted_group=np.repeat(np.expand_dims(logs['predicted_group'].values,1), 100, 1)
    
    cluster_actual_group=np.repeat(np.expand_dims(logs['group_2'].values,1), 100, 1)
    
    cmap_formation = colors.ListedColormap(formation_colors)  
    bounds_formation = [l for l in range(n_formation+1)]
    norm_formation = colors.BoundaryNorm(bounds_formation, cmap_formation.N)
    
    cmap_group = colors.ListedColormap(group_colors)
    bounds_group = [l for l in range(n_group+1)]
    norm_group = colors.BoundaryNorm(bounds_group, cmap_group.N)
    
    #ztop=logs.tvd.min(); zbot=logs.tvd.max()
        
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 12))

    im1=ax[0].imshow(cluster_predicted_formation, interpolation='none', aspect='auto',
                cmap=cmap_formation,vmin=0,vmax=37)#, norm = norm_formation)
    
    im2=ax[1].imshow(cluster_actual_formation, interpolation='none', aspect='auto',
                cmap=cmap_formation,vmin=0,vmax=37)#, norm = norm_formation)
    
    im3=ax[2].imshow(cluster_predicted_group, interpolation='none', aspect='auto',
                cmap=cmap_group,vmin=0,vmax=12)#, norm = norm_group)
    
    im4=ax[3].imshow(cluster_actual_group, interpolation='none', aspect='auto',
                cmap=cmap_group,vmin=0,vmax=12)#, norm = norm_group)
    
    ax[0].set_xlabel('Predicted formations')
    ax[1].set_xlabel('Actual formations')
    ax[2].set_xlabel('Predicted groups')
    ax[3].set_xlabel('Actual groups')
    
    ax[0].set_yticklabels([])
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    
    f.suptitle('Well: %s'%logs.iloc[0]['title'], fontsize=14,y=0.91)
    
    if save:
        plt.savefig(fig_dir+'prediction_'+model_name+'_'+'well_'+str(well_index)+'.png')
    plt.show()


# +
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

def plot_well_logs(df, well_index, formation_colors, save = False):
    
    df['group_2'] = df['group'].map(group_dictionary)
    
    logs = df.loc[df["title"] == df.title.unique()[well_index]]
    #logs = logs.sort_values(by='tvd')
    cmap_formation = colors.ListedColormap(formation_colors)  
    cmap_group = colors.ListedColormap(group_colors1)  
    
    ztop=logs.tvd.min(); zbot=logs.tvd.max()

    cluster=np.repeat(np.expand_dims(logs['formation_2'].values,1), 100, 1)
    cluster_2=np.repeat(np.expand_dims(logs['group_2'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=8, figsize=(12, 16))
    ax[0].plot(logs.gr, logs.tvd, '-g')
    ax[1].plot(logs.rdep, logs.tvd, '-')
    ax[2].plot(logs.rmed, logs.tvd, '-', color='r')
    ax[3].plot(logs.dt, logs.tvd, '-', color='0.5')
    ax[4].plot(logs.nphi, logs.tvd, '-', color='y')
    ax[5].plot(logs.rhob, logs.tvd, '-', color='c')

    im1=ax[6].imshow(cluster, interpolation='none', aspect='auto',
                cmap=cmap_formation,vmin=1,vmax=37)
    im2=ax[7].imshow(cluster_2, interpolation='none', aspect='auto',
                cmap=cmap_group,vmin=1,vmax=12)
    
    for i in range(len(ax)-2):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("gr")
    ax[0].set_xlim(logs.gr.min(),logs.gr.max()+10)
    ax[1].set_xlabel("rdep")
    ax[1].set_xlim(logs.rdep.min(),logs.rdep.max()+0.5)
    ax[2].set_xlabel("rmed")
    ax[2].set_xlim(logs.rmed.min(),logs.rmed.max()+0.5)
    ax[3].set_xlabel("dt")
    ax[3].set_xlim(logs.dt.min(),logs.dt.max()+0.5)
    ax[4].set_xlabel("nphi")
    ax[5].set_xlim(logs.nphi.min(),logs.nphi.max()+0.5)
    ax[5].set_xlabel("rhob")
    ax[5].set_xlim(logs.rhob.min(),logs.rhob.max()+0.5)
    ax[6].set_xlabel('Formations')
    ax[7].set_xlabel('Group')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([]);ax[4].set_yticklabels([])
    ax[5].set_yticklabels([]); ax[6].set_yticklabels([]); ax[7].set_yticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['title'], fontsize=14,y=0.91)
    
    if save:
        plt.savefig(fig_dir+'well_'+str(well_index)+'.01.png')
    plt.show()
