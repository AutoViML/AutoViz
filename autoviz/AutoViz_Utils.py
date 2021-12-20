############################################################################
#Copyright 2019 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#################################################################################################
import pandas as pd
import numpy as np
from pathlib import Path
import os
#### The warnings from Sklearn are so annoying that I have to shut it off ####
import warnings
warnings.filterwarnings("ignore")
def warn(*args, **kwargs):
    pass
warnings.warn = warn
########################################
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
####################################################################################
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from matplotlib import io
import io
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import re
import pdb
import pprint
import matplotlib
matplotlib.style.use('seaborn')
from itertools import cycle, combinations
from collections import defaultdict
import copy
import time
import sys
import random
import xlrd
import statsmodels
from io import BytesIO
import base64
from functools import reduce
import traceback
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
######## This is where we import HoloViews related libraries  #########
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import panel as pn
import panel.widgets as pnw
import holoviews.plotting.bokeh
from bokeh.resources import INLINE
######## This is where we store the image data in a dictionary with a list of images #########
def save_image_data(fig, chart_format, plot_name, depVar, mk_dir, additional=''):
    if not os.path.isdir(mk_dir):
        os.mkdir(mk_dir)
    if additional == '':
        filename = os.path.join(mk_dir,plot_name+"."+chart_format)
    else:
        filename = os.path.join(mk_dir,plot_name+additional+"."+chart_format)
    ##################################################################################
    if chart_format == 'svg':
        ###### You have to add these lines to each function that creates charts currently ##
        imgdata = io.StringIO()
        fig.savefig(filename, dpi='figure', format=chart_format)
        imgdata.seek(0)
        svg_data = imgdata.getvalue()
        return svg_data
    else:
        ### You have to do it slightly differently for PNG and JPEG formats
        imgdata = BytesIO()
        fig.savefig(filename,format=chart_format, dpi='figure')
        #fig.savefig(imgdata, format=chart_format, bbox_inches='tight', pad_inches=0.0)
        imgdata.seek(0)
        figdata_png = base64.b64encode(imgdata.getvalue())
        return figdata_png

def save_html_data(hv_all, chart_format, plot_name, mk_dir, additional=''):
    print('Saving %s in HTML format' %(plot_name+additional))
    if not os.path.isdir(mk_dir):
        os.mkdir(mk_dir)
    if additional == '':
        filename = os.path.join(mk_dir,plot_name+"."+chart_format)
    else:
        filename = os.path.join(mk_dir,plot_name+additional+"."+chart_format)
    ## it is amazing you can save interactive plots ##
    ## You don't need the resources = INLINE since it would consume too much space in HTML plots
    #pn.panel(hv_all).save(filename, embed=True, resources=INLINE) 
    pn.panel(hv_all).save(filename, embed=True)

#### This module analyzes a dependent Variable and finds out whether it is a
#### Regression or Classification type problem
def analyze_problem_type(train, target, verbose=0) :

    target = copy.deepcopy(target)
    cat_limit = 30 ### this determines the number of categories to name integers as classification ##
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    
    if isinstance(target, str):
        target = [target]
    if len(target) == 1:
        targ = target[0]
    else:
        targ = target[0]
    ####  This is where you detect what kind of problem it is #################
    if  train[targ].dtype in ['int64', 'int32','int16']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= cat_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    elif  train[targ].dtype in ['float16','float32','float64']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= float_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    elif train[targ].dtype == bool:
        model_class = 'Binary_Classification'
    else:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        else:
            model_class = 'Multi_Classification'
    ########### print this for the start of next step ###########
    if verbose <= 2:
        print('''\n################ %s VISUALIZATION Started #####################''' %model_class)
    return model_class
#################################################################################
# Pivot Tables are generally meant for Categorical Variables on the axes
# and a Numeric Column (typically the Dep Var) as the "Value" aggregated by Sum.
# Let's do some pivot tables to capture some meaningful insights
import random
def draw_pivot_tables(dft,cats,nums,problem_type,verbose,chart_format,depVar='', classes=None, mk_dir=None):
    plot_name = 'Bar_Plots_Pivots'
    cats = list(set(cats))
    dft = dft[:]
    cols = 2
    cmap = plt.get_cmap('jet')
    #### For some reason, the cmap colors are not working #########################
    colors = cmap(np.linspace(0, 1, len(cats)))
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgkbyr')
    #colormaps = ['summer', 'rainbow','viridis','inferno','magma','jet','plasma']
    colormaps = ['Greys','Blues','Greens','GnBu','PuBu',
                    'YlGnBu','PuBuGn','BuGn','YlGn']
    #colormaps = ['Purples','Oranges','Reds','YlOrBr',
    #                'YlOrRd','OrRd','PuRd','RdPu','BuPu',]
    N = len(cats)
    if N==0:
        print('No categorical or boolean vars in data set. Hence no pivot plots...')
        return None
    noplots = copy.deepcopy(N)
    #### You can set the number of subplots per row and the number of categories to display here cols = 2
    displaylimit = 20
    categorylimit = 10
    imgdata_list = []
    width_size = 15
    height_size = 5
    stringlimit = 20
    combos = combinations(cats, 2)
    N = len(cats)
    if N <= 1:
        ### if there are not many categorical variables, there is nothing to plot
        return imgdata_list
    if len(nums) == 0:
        ### if there are no numeric variables, there is nothing to plot
        return imgdata_list
    if not depVar is None or not depVar=='' or not depVar==[] :
        ###########  This works equally well for classification as well as Regression ###
        lst=[]
        noplots=int((N**2-N)/2)
        dicti = {}
        counter = 1
        cols = 2
        if noplots%cols == 0:
            if noplots == 0:
                rows = 1
            else:
                rows = noplots/cols
        else:
            rows = (noplots/cols)+1
        #fig = plt.figure(figsize=(min(20,N*10),rows*5))
        fig = plt.figure()
        if cols < 2:
            fig.set_size_inches(min(15,8),rows*height_size)
            fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
            fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
        else:
            fig.set_size_inches(min(cols*10,20),rows*height_size)
            fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
            fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
        for (var1, var2) in combos:
            color1 = random.choice(colormaps)
            data = pd.DataFrame(dicti)
            x=dft[var1]
            y=dft[var2]
            ax1 = fig.add_subplot(rows,cols,counter)
            nocats = min(categorylimit,dft[var1].nunique())
            nocats1 = min(categorylimit,dft[var2].nunique())
            if dft[depVar].dtype==object or dft[depVar].dtype==bool:
                dft[depVar] = dft[depVar].factorize()[0]
            data = pd.pivot_table(dft,values=depVar,index=var1, columns=var2).head(nocats)
            data = data[data.columns[:nocats1]] #### make sure you don't print more than 10 rows of data
            data.plot(kind='bar',ax=ax1,colormap=color1)
            ax1.set_xlabel(var1)
            ax1.set_ylabel(depVar)
            if dft[var1].dtype == object or str(dft[var1].dtype) == 'category':
                labels = data.index.str[:stringlimit].tolist()
            else:
                labels = data.index.tolist()
            ax1.set_xticklabels(labels,fontdict={'fontsize':10}, rotation = 45, ha="right")
            ax1.legend(fontsize="medium")
            ax1.set_title('%s (Mean) by %s and %s' %(depVar,var1,var2),fontsize=12)
            counter += 1
        fig.tight_layout()
        fig.suptitle('Pivot Tables of each Continuous var by 2 Categoricals', fontsize=15,y=1.01);
    else:
        print('No pivot tables plotted since no dependent variable given as input')
    image_count = 0
    if verbose == 2:
        imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name, depVar, mk_dir))
        image_count += 1
    if verbose <= 1:
        plt.show();
    ####### End of Pivot Plotting #############################
    return imgdata_list

# In[ ]:
# SCATTER PLOTS ARE USEFUL FOR COMPARING NUMERIC VARIABLES
def draw_scatters(dfin,nums,verbose,chart_format,problem_type,dep=None, classes=None, lowess=False, mk_dir=None):
    plot_name = 'Scatter_Plots'
    dft = dfin[:]
    ##### we are going to modify dfin and classes, so we are making copies to make changes
    classes = copy.deepcopy(classes)
    colortext = 'brymcgkbyrcmgkbyrcmgkbyrcmgkbyr'
    if len(classes) == 0:
        leng = len(nums)
    else:
        leng = len(classes)
    colors = cycle(colortext[:leng])
    #imgdata_list = defaultdict(list)
    imgdata_list = []
    if dfin.shape[0] >= 10000 or lowess == False:
        lowess = False
        x_est = None
        transparent = 0.6
        bubble_size = 80
    else:
        if verbose <= 1:
            print('Using Lowess Smoothing. This might take a few minutes for large data sets...')
        lowess = True
        x_est = None
        transparent = 0.6
        bubble_size = 100
    if verbose <= 1:
        x_est = np.mean
    N = len(nums)
    cols = 2
    width_size = 15
    height_size = 4
    if dep == None or dep == '':
        ### when there is no dependent variable, you can't plot anything in scatters here ###
        return None
    elif problem_type == 'Regression':
        image_count = 0
        ####### This is a Regression Problem so it requires 2 steps ####
        ####### First, plot every Independent variable against the Dependent Variable ###
        noplots = len(nums)
        rows = int((noplots/cols)+0.99)
        fig = plt.figure(figsize=(width_size,rows*height_size))
        for num, plotcounter, color_val in zip(nums, range(1,noplots+1), colors):
            ### Be very careful with the next line. It should be singular "subplot" ##
            ##### Otherwise, if you use the plural version "subplots" it has a different meaning!
            plt.subplot(rows,cols,plotcounter)
            if lowess:
                sns.regplot(x=dft[num], y = dft[dep], lowess=lowess, color=color_val, ax=plt.gca())
            else:
                sns.scatterplot(x=dft[num], y=dft[dep], ax=plt.gca(), palette='dark',color=color_val)
            plt.xlabel(num)
            plt.ylabel(dep)
        fig.suptitle('Scatter Plot of each Continuous Variable vs Target',fontsize=15,y=1.01)
        fig.tight_layout();
        if verbose <= 1:
            plt.show();
        #### Keep it at the figure level###
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name, dep, mk_dir))
            image_count += 1
    else:
        ####### This is a Classification Problem #### You need to plot a strip plot ####
        ####### First, Plot each Continuous variable against the Target Variable ###
        if len(dft) < 1000:
            jitter = 0.05
        else:
            jitter = 0.5
        image_count = 0
        noplots = len(nums)
        rows = int((noplots/cols)+0.99)
        ### Be very careful with the next line. we have used the singular "subplot" ##
        fig = plt.figure(figsize=(width_size,rows*height_size))
        for num, plotc, color_val in zip(nums, range(1,noplots+1),colors):
            ####Strip plots are meant for categorical plots so x axis must always be depVar ##
            plt.subplot(rows,cols,plotc)
            sns.stripplot(x=dft[dep], y=dft[num], ax=plt.gca(), jitter=jitter)
        plt.suptitle('Scatter Plot of Continuous Variable vs Target (jitter=%0.2f)' %jitter, fontsize=15,y=1.01)
        fig.tight_layout();
        if verbose <= 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name, dep, mk_dir))
            image_count += 1
    ####### End of Scatter Plots ######
    return imgdata_list

# PAIR SCATTER PLOTS ARE NEEDED ONLY FOR CLASSIFICATION PROBLEMS IN NUMERIC VARIABLES
def draw_pair_scatters(dfin,nums,problem_type, verbose,chart_format, dep=None, classes=None, lowess=False, mk_dir=None):
    """
    ### This is where you plot a pair-wise scatter plot of Independent Variables against each other####
    """
    plot_name = 'Pair_Scatter_Plots'
    dft = dfin[:]
    if len(nums) <= 1:
        return
    classes = copy.deepcopy(classes)
    cols = 2
    colortext = 'brymcgkbyrcmgkbyrcmgkbyrcmgkbyr'
    colors = cycle(colortext)
    imgdata_list = list()
    width_size = 15
    height_size = 4
    N = len(nums)
    if dfin.shape[0] >= 10000 or lowess == False:
        x_est = None
        transparent =0.7
        bubble_size = 80
    elif lowess:
        print('Using Lowess Smoothing. This might take a few minutes for large data sets...')
        x_est = None
        transparent =0.7
        bubble_size = 100
    else:
        x_est = None
        transparent =0.7
        bubble_size = 100
    if verbose <= 1:
        x_est = np.mean
    if problem_type == 'Regression' or problem_type == 'Clustering':
        image_count = 0
        ### Second, plot a pair-wise scatter plot of Independent Variables against each other####
        combos = combinations(nums, 2)
        noplots = int((N**2-N)/2)
        print('Number of All Scatter Plots = %d' %(noplots+N))
        rows = int((noplots/cols)+0.99)
        fig = plt.figure(figsize=(width_size,rows*height_size))
        for (var1,var2), plotcounter,color_val in zip(combos, range(1,noplots+1),colors):
            ### Be very careful with the next line. It should be singular "subplot" ##
            ##### Otherwise, if you use the plural version "subplots" it has a different meaning!
            plt.subplot(rows,cols,plotcounter)
            if lowess:
                sns.regplot(x=dft[var1], y=dft[var2], lowess=lowess, color=color_val, ax=plt.gca())
            else:
                sns.scatterplot(x=dft[var1], y=dft[var2], ax=plt.gca(), palette='dark',color=color_val)
            plt.xlabel(var1)
            plt.ylabel(var2)
        fig.suptitle('Pair-wise Scatter Plot of all Continuous Variables', fontsize=15,y=1.01)
        fig.tight_layout();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name, dep, mk_dir))
            image_count += 1
        if verbose <= 1:
            plt.show();
    else:
        ########## This is for Classification problems ##########
        if len(classes) <= 1:
            leng = 1
        else:
            leng = len(classes)
        colors = cycle(colortext[:leng])
        image_count = 0
        #cmap = plt.get_cmap('gnuplot')
        #cmap = plt.get_cmap('Set1')
        cmap = plt.get_cmap('Paired')
        combos = combinations(nums, 2)
        combos_cycle = cycle(combos)
        noplots = int((N**2-N)/2)
        print('Total Number of Scatter Plots = %d' %(noplots+N))
        rows = int((noplots/cols)+0.99)
        fig = plt.figure(figsize=(width_size,rows*height_size))
        ### Be very careful with the next line. we have used the plural "subplots" ##
        ## In this case, you have ax as an array and you have to use (row,col) to get each ax!
        target_vars = dft[dep].unique()
        number = len(target_vars)
        #colors = [cmap(i) for i in np.linspace(0, 1, number)]
        for (var1,var2), plotc in zip(combos, range(1,noplots+1)):
            for target_var, color_val, class_label in zip(target_vars, colors, classes):
                #Fix color in all scatter plots for each class the same using this trick
                color_array = np.empty(0)
                value = dft[dep]==target_var
                dft['color'] = np.where(value==True, color_val, 'r')
                color_array = np.hstack((color_array, dft[dft['color']==color_val]['color'].values))
                plt.subplot(rows, cols, plotc)
                plt.scatter(x=dft.loc[dft[dep]==target_var][var1], y=dft.loc[dft[dep]==target_var][var2],
                             label=class_label, color=color_val, alpha=transparent)
                plt.xlabel(var1)
                plt.ylabel(var2)
                plt.legend()
        fig.suptitle('Pair-wise Scatter Plot of all Continuous Variables',fontsize=15,y=1.01)
        #fig.tight_layout();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name, dep, mk_dir))
            image_count += 1
        if verbose <= 1:
            plt.show();
    ####### End of Pair Scatter Plots ######
    return imgdata_list

#Bar Plots are for 2 Categoricals and One Numeric (usually Dep Var)
def plot_fast_average_num_by_cat(dft, cats, num_vars, verbose=0,kind="bar"):
    """
    Great way to plot continuous variables fast grouped by a categorical variable. Just sent them in and it will take care of the rest!
    """
    chunksize = 20
    stringlimit = 20
    col = 2
    width_size = 15
    height_size = 4
    N = int(len(num_vars)*len(cats))
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    if N % 2 == 0:
        row = N//col
    else:
        row = int(N//col + 1)
    fig = plt.figure()
    if kind == 'bar':
        fig.suptitle('Bar plots for each Continuous by each Categorical variable', fontsize=15,y=1.01)
    else:
        fig.suptitle('Time Series plots for all date-time vars %s' %cats, fontsize=15,y=1.01)
    if col < 2:
        fig.set_size_inches(min(15,8),row*5)
        fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
    else:
        fig.set_size_inches(min(col*10,20),row*5)
        fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
    counter = 1
    for cat in cats:
        for each_conti in num_vars:
            color3 = next(colors)
            try:
                ax1 = plt.subplot(row, col, counter)
                if kind == "bar":
                    data = dft.groupby(cat)[each_conti].mean().sort_values(
                            ascending=False).head(chunksize)
                    data.plot(kind=kind,ax=ax1,color=color3)
                elif kind == "line":
                    data = dft.groupby(cat)[each_conti].mean().sort_index(
                            ascending=True).head(chunksize)
                    data.plot(kind=kind,ax=ax1,color=color3)
                if dft[cat].dtype == object or str(dft[cat].dtype) == 'category':
                    labels = data.index.str[:stringlimit].tolist()
                else:
                    labels = data.index.tolist()
                ax1.set_xlabel("")
                ax1.set_xticklabels(labels,fontdict={'fontsize':9}, rotation = 45, ha="right")
                ax1.set_title('Average %s by %s (Top %d)' %(each_conti,cat,chunksize))
                counter += 1
            except:
                ax1.set_title('No plot as %s is not numeric' %each_conti)
                counter += 1
    if verbose <= 1:
        plt.show()
    if verbose == 2:
        return fig
################# The barplots module below calls the plot_fast_average_num_by_cat module above ###
def draw_barplots(dft,cats,conti,problem_type,verbose,chart_format,dep='', classes=None, mk_dir=None):
    cats = cats[:]
    conti = conti[:]
    plot_name = 'Bar_Plots'
    #### Category limit within a variable ###
    #### Remove Floating Point Categorical Vars from this list since they Error when Bar Plots are drawn
    cats = [x for x in cats if dft[x].dtype != float]
    dft = dft[:]
    N = len(cats)
    if len(cats) == 0 or len(conti) == 0:
        print('No categorical or numeric vars in data set. Hence no bar charts.')
        return None
    cmap = plt.get_cmap('jet')
    ### Not sure why the cmap doesn't work and gives an error in some cases #################
    colors = cmap(np.linspace(0, 1, len(conti)))
    colors = cycle('gkbyrcmgkbyrcmgkbyrcmgkbyr')
    colormaps = ['plasma','viridis','inferno','magma']
    imgdata_list = list()
    cat_limit = 10
    conti = list_difference(conti,dep)
    #### Make sure that you plot charts for the depVar as well by including it #######
    if problem_type == 'Regression':
        conti.append(dep)
    elif problem_type.endswith('Classification'):
        cats.append(dep)
    else:
        ### Since there is no dependent variable in clustering there is nothing to add dep to.
        pass
    chunksize = 20
    ########## This is for Regression Problems only ######
    image_count = 0
    figx = plot_fast_average_num_by_cat(dft, cats, conti, verbose)
    if verbose == 2:
        imgdata_list.append(save_image_data(figx, chart_format,
                            plot_name, dep, mk_dir))
        image_count += 1
    return imgdata_list
############## End of Bar Plotting ##########################################
##### Draw a Heatmap using Pearson Correlation #########################################
def draw_heatmap(dft, conti, verbose,chart_format,datevars=[], dep=None,
                                    modeltype='Regression',classes=None, mk_dir=None):
    ### Test if this is a time series data set, then differene the continuous vars to find
    ###  if they have true correlation to Dependent Var. Otherwise, leave them as is
    plot_name = 'Heat_Maps'
    width_size = 3
    height_size = 2
    timeseries_flag = False
    if len(conti) <= 1:
        return
    if isinstance(dft.index, pd.DatetimeIndex) :
        dft = dft[:]
        timeseries_flag = True
        pass
    elif len(datevars) > 0:
        dft = dft[:]
        try:
            dft.index = pd.to_datetime(dft.pop(datevars[0]),infer_datetime_format=True)
            timeseries_flag = True
        except:
            if verbose >= 1 and len(datevars) > 0:
                print('No date vars could be found or %s could not be indexed.' %datevars)
            elif verbose >= 1 and len(datevars) == 0:
                print('No date vars could be found in data set')
            timeseries_flag = False
    # Add a column: the color depends on target variable but you can use whatever function
    imgdata_list = list()
    if modeltype.endswith('Classification'):
        ########## This is for Classification problems only ###########
        if dft[dep].dtype == object or dft[dep].dtype == np.int64:
            dft[dep] = dft[dep].factorize()[0]
        image_count = 0
        N = len(conti)
        target_vars = dft[dep].unique()
        fig = plt.figure(figsize=(min(N*width_size,20),min(N*height_size,20)))
        if timeseries_flag:
            fig.suptitle('Time Series: Heatmap of all Differenced Continuous vars for target = %s' %dep, fontsize=15,y=1.01)
        else:
            fig.suptitle('Heatmap of all Continuous Variables for target = %s' %dep, fontsize=15,y=1.01)
        plotc = 1
        #rows = len(target_vars)
        rows = 1
        cols = 1
        if timeseries_flag:
            dft_target = dft[[dep]+conti].diff()
        else:
            dft_target = dft[:]
        dft_target[dep] = dft[dep].values
        corr = dft_target.corr()
        plt.subplot(rows, cols, plotc)
        ax1 = plt.gca()
        sns.heatmap(corr, annot=True,ax=ax1)
        plotc += 1
        fig.tight_layout();
        if verbose <= 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name, dep, mk_dir))
            image_count += 1
    else:
        ### This is for Regression and None Dep variable problems only ##
        image_count = 0
        if dep == None or dep == '':
            pass
        else:
            conti += [dep]
        dft_target = dft[conti]
        if timeseries_flag:
            dft_target = dft_target.diff().dropna()
        else:
            dft_target = dft_target[:]
        N = len(conti)
        fig = plt.figure(figsize=(min(20,N*width_size),min(20,N*height_size)))
        corr = dft_target.corr()
        sns.heatmap(corr, annot=True)
        if timeseries_flag:
            fig.suptitle('Time Series Data: Heatmap of Differenced Continuous vars including target = %s' %dep, fontsize=15,y=1.01)
        else:
            fig.suptitle('Heatmap of all Continuous Variables including target = %s' %dep,fontsize=15,y=1.01)
        fig.tight_layout();
        if verbose <= 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name, dep, mk_dir))
            image_count += 1
    return imgdata_list
    ############# End of Heat Maps ##############

##### Draw the Distribution of each variable using Distplot
##### Must do this only for Continuous Variables
from scipy.stats import probplot,skew
def draw_distplot(dft, cat_bools, conti, verbose,chart_format,problem_type,dep=None, classes=None, mk_dir=None):
    cats = find_remove_duplicates(cat_bools) ### first make sure there are no duplicates in this ###
    copy_cats = copy.deepcopy(cats)
    plot_name = 'Dist_Plots'
    #### Since we are making changes to dft and classes, we will be making copies of it here
    conti = list(set(conti))
    dft = dft[:]
    classes = copy.deepcopy(classes)
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    imgdata_list = list()
    width_size = 15  #### this is to control the width of chart as well as number of categories to display
    height_size = 5
    gap = 0.4 #### This controls the space between rows  ######
    if dep==None or dep=='' or problem_type == 'Regression':
        image_count = 0
        transparent = 0.7
        ######### This is for cases where there is No Target or Dependent Variable ########
        if problem_type == 'Regression':
            if isinstance(dep,list):
                conti += dep
            else:
                conti += [dep]
        ### Be very careful with the next line. we have used the plural "subplots" ##
        ## In this case, you have ax as an array and you have to use (row,col) to get each ax!
        ########## This is where you insert the logic for displots ##############
        sns.color_palette("Set1")
        ##### First draw all the numeric variables in row after row #############
        if not len(conti) == 0:
            cols = 3
            rows = len(conti)
            fig, axes = plt.subplots(rows, cols, figsize=(width_size,rows*height_size))
            k = 1
            for each_conti in conti:
                color1 = next(colors)
                ax1 = plt.subplot(rows, cols, k)
                sns.distplot(dft[each_conti],kde=False, ax=ax1, color=color1)
                k += 1
                ax2 = plt.subplot(rows, cols, k)
                sns.boxplot(dft[each_conti], ax=ax2, color=color1)
                k += 1
                ax3 = plt.subplot(rows, cols, k)
                probplot(dft[each_conti], plot=ax3)
                k += 1
                skew_val=round(dft[each_conti].skew(), 1)
                ax2.set_yticklabels([])
                ax2.set_yticks([])
                ax1.set_title(each_conti + " | Distplot")
                ax2.set_title(each_conti + " | Boxplot")
                ax3.set_title(each_conti + " | Probability Plot - Skew: "+str(skew_val))
            ###### Save the plots to disk if verbose = 2 ############
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, chart_format,
                                plot_name+'_Numeric', dep, mk_dir))
                image_count += 1
        #####  Now draw each of the categorical variable distributions in each subplot ####
        if not len(cats) == 0:
            cols = 2
            noplots = len(cats)
            rows = int((noplots/cols)+0.99 )
            k = 0
            fig = plt.figure(figsize=(width_size,rows*height_size))
            fig.subplots_adjust(hspace=gap) ### This controls the space betwen rows
            for each_cat in copy_cats:
                color2 = next(colors)
                ax1 = plt.subplot(rows, cols, k+1)
                kwds = {"rotation": 45, "ha":"right"}
                labels = dft[each_cat].value_counts()[:width_size].index.tolist()
                dft[each_cat].value_counts()[:width_size].plot(kind='bar', color=color2,
                                            ax=ax1,label='%s' %each_cat)
                ax1.set_xticklabels(labels,**kwds);
                ax1.set_title('Distribution of %s (top %d categories only)' %(each_cat,width_size))
            fig.tight_layout();
            ########## This is where you end the logic for distplots ################
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, chart_format,
                                plot_name+'_Cats', dep, mk_dir))
                image_count += 1
            fig.suptitle('Histograms (KDE plots) of all Continuous Variables', fontsize=12,y=1.01)
            if verbose <= 1:
                plt.show();
    else:
        ######### This is for Classification problems only ########
        #### Now you can draw both object and numeric variables using same conti_
        conti = conti + cats
        cols = 2
        image_count = 0
        transparent = 0.7
        noplots = len(conti)
        binsize = 30
        k = 0
        rows = int((noplots/cols)+0.99 )
        ### Be very careful with the next line. we have used the plural "subplots" ##
        ## In this case, you have ax as an array and you have to use (row,col) to get each ax!
        fig = plt.figure(figsize=(width_size,rows*height_size))
        target_vars = dft[dep].unique()
        if type(classes[0])==int:
            classes = [str(x) for x in classes]
        label_limit = len(target_vars)
        legend_flag = 1
        for each_conti,k in zip(conti,range(len(conti))):
            if dft[each_conti].isnull().sum() > 0:
                dft[each_conti].fillna(0, inplace=True)
            plt.subplot(rows, cols, k+1)
            ax1 = plt.gca()
            if dft[each_conti].dtype==object:
                kwds = {"rotation": 45, "ha":"right"}
                labels = dft[each_conti].value_counts()[:width_size].index.tolist()
                conti_df = dft[[dep,each_conti]].groupby([dep,each_conti]).size().nlargest(width_size).reset_index(name='Values')
                pivot_df = conti_df.pivot(index=each_conti, columns=dep, values='Values')
                row_ticks = dft[dep].unique().tolist()
                color_list = []
                for i in range(len(row_ticks)):
                    color_list.append(next(colors))
                #print('color list = %s' %color_list)
                pivot_df.loc[:,row_ticks].plot.bar(stacked=True, color=color_list, ax=ax1)
                #dft[each_conti].value_counts()[:width_size].plot(kind='bar',ax=ax1,
                #                    label=class_label)
                #ax1.set_xticklabels(labels,**kwds);
                ax1.set_title('Distribution of %s (top %d categories only)' %(each_conti,width_size))
            else:
                for target_var, color2, class_label in zip(target_vars,colors,classes):
                    try:
                        if legend_flag <= label_limit:
                            sns.distplot(dft.loc[dft[dep]==target_var][each_conti],
                                hist=False, kde=True,
                            #dft.ix[dft[dep]==target_var][each_conti].hist(
                                bins=binsize, ax= ax1,
                                label=target_var, color=color2)
                            ax1.set_title('Distribution of %s' %each_conti)
                            legend_flag += 1
                        else:
                            sns.distplot(dft.loc[dft[dep]==target_var][each_conti],bins=binsize, ax= ax1,
                            label=target_var, hist=False, kde=True,
                            color=color2)
                            legend_flag += 1
                            ax1.set_title('Normed Histogram of %s' %each_conti)
                    except:
                        pass
            ax1.legend(loc='best')
            k += 1
        fig.tight_layout();
        if verbose <= 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name+'_Numerics', dep, mk_dir))
            image_count += 1
        fig.suptitle('Histograms (KDE plots) of all Continuous Variables', fontsize=12,y=1.01)
        ###### Now draw the distribution of the target variable in Classification only ####
        #####  Now draw each of the categorical variable distributions in each subplot ####
        ############################################################################
        if problem_type.endswith('Classification'):
            col = 2
            row = 1
            fig, (ax1,ax2) = plt.subplots(row, col)
            fig.set_figheight(5)
            fig.set_figwidth(15)
            fig.suptitle('%s : Distribution of Target Variable' %dep, fontsize=12)
            #fig.subplots_adjust(hspace=0.3) ### This controls the space betwen rows
            #fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
            ###### Precentage Distribution is first #################
            dft[dep].value_counts(1).plot(ax=ax1,kind='bar')
            if dft[dep].dtype == object:
                dft[dep] = dft[dep].factorize()[0]
            for p in ax1.patches:
                ax1.annotate(str(round(p.get_height(),2)), (round(p.get_x()*1.01,2), round(p.get_height()*1.01,2)))
            ax1.set_xticks(dft[dep].unique().tolist())
            ax1.set_xticklabels(classes, rotation = 45, ha="right", fontsize=9)
            ax1.set_title('Percentage Distribution of Target = %s' %dep, fontsize=10, y=1.05)
            #### Freq Distribution is next ###########################
            dft[dep].value_counts().plot(ax=ax2,kind='bar')
            for p in ax2.patches:
                ax2.annotate(str(round(p.get_height(),2)), (round(p.get_x()*1.01,2), round(p.get_height()*1.01,2)))
            ax2.set_xticks(dft[dep].unique().tolist())
            ax2.set_xticklabels(classes, rotation = 45, ha="right", fontsize=9)
            ax2.set_title('Freq Distribution of Target Variable = %s' %dep,  fontsize=12)
        elif problem_type == 'Regression':
            ############################################################################
            width_size = 5
            height_size = 5
            fig = plt.figure(figsize=(width_size,height_size))
            dft[dep].plot(kind='hist')
            fig.suptitle('%s : Distribution of Target Variable' %dep, fontsize=12)
            fig.tight_layout();
        else:
            return imgdata_list
        if verbose <= 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name+'_target', dep, mk_dir))
            image_count += 1
    ####### End of Distplots ###########
    return imgdata_list

##### Standardize all the variables in One step. But be careful !
#### All the variables must be numeric for this to work !!
def draw_violinplot(df, dep, nums,verbose,chart_format, modeltype='Regression', mk_dir=None):
    plot_name = 'Violin_Plots'
    df = df[:]
    number_in_each_row = 8
    imgdata_list = list()
    width_size = 15
    height_size = 4
    if type(dep) == str:
        othernums = [x for x in nums if x not in [dep]]
    else:
        othernums = [x for x in nums if x not in dep]
    if modeltype == 'Regression' or dep == None or dep == '':
        image_count = 0
        if modeltype == 'Regression':
            nums = nums + [dep]
        numb = len(nums)
        if numb > number_in_each_row:
            rows = int(numb/number_in_each_row)+1
        else:
            rows = 1
        plot_index = 0
        for row in range(rows):
            plot_index += 1
            first_10 = number_in_each_row*row
            next_10 = first_10 + number_in_each_row
            num_10 = nums[first_10:next_10]
            df10 = df[num_10]
            df_norm = (df10 - df10.mean())/df10.std()
            if numb <= 5:
                fig = plt.figure(figsize=(min(width_size*len(num_10),width_size),min(height_size,height_size*len(num_10))))
            else:
                fig = plt.figure(figsize=(min(width_size*len(num_10),width_size),min(height_size,height_size*len(num_10))))
            ax = fig.gca()
            #ax.set_xticklabels (df.columns, tolist(), size=10)
            sns.violinplot(data=df_norm, orient='v', fliersize=5, scale='width',
                linewidth=3, notch=False, saturations=0.5, ax=ax, inner='box')
            fig.suptitle('Violin Plot of all Continuous Variables', fontsize=15)
            fig.tight_layout();
            if verbose <= 1:
                plt.show();
            if verbose == 2:
                additional = '_'+str(plot_index)+'_'
                imgdata_list.append(save_image_data(fig, chart_format,
                                plot_name, dep, mk_dir, additional))
                image_count += 1
    else :
        plot_name = "Box_Plots"
        ###### This is for Classification problems only ##########################
        image_count = 0
        classes = df[dep].factorize()[1].tolist()
        ######################### Add Box plots here ##################################
        numb = len(nums)
        target_vars = df[dep].unique()
        if len(othernums) >= 1:
            width_size = 15
            height_size = 7
            count = 0
            data = pd.DataFrame(index=df.index)
            cols = 2
            noplots = len(nums)
            rows = int((noplots/cols)+0.99 )
            fig = plt.figure(figsize=(width_size,rows*height_size))
            for col in nums:
                ax = plt.subplot(rows,cols,count+1)
                for targetvar in target_vars:
                    data[targetvar] = np.nan
                    mask = df[dep]==targetvar
                    data.loc[mask,targetvar] = df.loc[mask,col]
                ax = sns.boxplot(data=data, orient='v', fliersize=5, ax=ax,
                        linewidth=3, notch=False, saturation=0.5, showfliers=False)
                ax.set_title('%s for each %s' %(col,dep))
                count += 1
            fig.suptitle('Box Plots without Outliers shown',  fontsize=15)
            fig.tight_layout();
            if verbose <= 1:
                plt.show();
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, chart_format,
                                plot_name, dep, mk_dir))
                image_count += 1
        #########################################
    return imgdata_list
    ########## End of Violin Plots #########

#### Drawing Date Variables is very important in Time Series data
import copy
def draw_date_vars(dfx,dep,datevars, num_vars,verbose, chart_format, modeltype='Regression', mk_dir=None):
    dfx = copy.deepcopy(dfx) ## use this to preserve the original dataframe
    df =  copy.deepcopy(dfx) #### use this for making it into a datetime index etc...
    plot_name = 'Time_Series_Plots'
    #### Now you want to display 2 variables at a time to see how they change over time
    ### Don't change the number of cols since you will have to change rows formula as well
    gap = 0.3 ### adjusts the gap between rows in multiple rows of charts
    imgdata_list = list()
    image_count = 0
    N = len(num_vars)
    chunksize = 20
    if N < 1 or len(datevars) == 0:
        #### If there are no numeric variables, nothing to plot here ######
        return imgdata_list
    else:
        width_size = 15
        height_size = 5
    if isinstance(df.index, pd.DatetimeIndex) :
        pass
    elif len(datevars) > 0:
        try:
            ts_column = datevars[0]
            ### if we have already found that it was a date time var, then leave it as it is. Thats good enough!
            date_items = df[ts_column].apply(str).apply(len).values
            date_4_digit = all(date_items[0] == item for item in date_items) ### this checks for 4 digits date
            #### In some cases, date time variable is a year like 1999 (4-digit), this must be translated correctly
            if date_4_digit:
                if date_items[0] == 4:
                    ### If it is just a year variable alone, you should leave it as just a year!
                    df[ts_column] = df[ts_column].map(lambda x: pd.to_datetime(x,format='%Y', errors='coerce')).values
                else:
                    ### if it is not a year alone, then convert it into a date time variable
                    if df[col].min() > 1900 or df[col].max() < 2100:
                        df[ts_column] = df[ts_column].map(lambda x: '0101'+str(x) if len(str(x)) == 4 else x)
                        df[ts_column] = pd.to_datetime(df[ts_column], format='%m%d%Y', errors='coerce')
                    else:
                        print('%s could not be indexed. Could not draw date_vars.' %col)
                        return imgdata_list
            else:
                df[ts_column] = pd.to_datetime(df[ts_column], infer_datetime_format=True, errors='coerce')
            ##### Now set the column to be the date - time index
            df.index = df.pop(ts_column) #### This is where we set the date time column as the index ######
        except:
            print('%s could not be indexed. Could not draw date_vars.' %col)
            return imgdata_list
    ####### Draw the time series for Regression and DepVar
    width_size = 15
    height_size = 4
    cols = 2
    if modeltype == 'Regression':
        gap=0.5
        rows = int(len(num_vars)/cols+0.99)
        fig,ax = plt.subplots(figsize=(width_size,rows*height_size))
        fig.subplots_adjust(hspace=gap) ### This controls the space betwen rows
        df.groupby(ts_column).mean().plot(subplots=True,ax=ax,layout=(rows,cols))
        fig.suptitle('Time Series Plot for each Continuous Variable by %s' %ts_column, fontsize=15,y=1.01)
    elif modeltype == 'Clustering':
        kind = 'line' #### you can change this to plot any kind of time series plot you want
        image_count = 0
        combos = combinations(num_vars, 2)
        combs = copy.deepcopy(combos)
        noplots = int((N**2-N)/2)
        rows = int((noplots/cols)+0.99)
        counter = 1
        fig = plt.figure(figsize=(width_size,rows*height_size))
        fig.subplots_adjust(hspace=gap) ### This controls the space betwen rows
        try:
            for (var1,var2) in combos:
                plt.subplot(rows,cols,counter)
                ax1 = plt.gca()
                df[var1].plot(kind=kind, secondary_y=True, label=var1, ax=ax1)
                df[var2].plot(kind=kind, title=var2 +' (left_axis) vs. ' + var1+' (right_axis)', ax=ax1)
                plt.legend(loc='best')
                counter += 1
                fig.suptitle('Time Series Plot by %s: Pairwise Continuous Variables' %ts_column, fontsize=15,y=1.01)
        except:
            plt.close('all')
            fig = plot_fast_average_num_by_cat(dfx, datevars, num_vars, verbose,kind="line")
    else:
        ######## This is for Classification problems only ####
        kind = 'line' ### you can decide what kind of plots you want to show here ####
        image_count = 0
        target_vars = df[dep].factorize()[1].tolist()
        #classes = copy.deepcopy(classes)
        ##### Now separate out the drawing of time series data by the number of classes ###
        colors = cycle('gkbyrcmgkbyrcmgkbyrcmgkbyr')
        classes = df[dep].unique()
        if type(classes[0])==int or type(classes[0])==float:
            classes = [str(x) for x in classes]
        cols = 2
        count = 0
        combos = combinations(num_vars, 2)
        combs = copy.deepcopy(combos)
        noplots = len(target_vars)
        rows = int((noplots/cols)+0.99)
        fig = plt.figure(figsize=(width_size,rows*height_size))
        fig.subplots_adjust(hspace=gap) ### This controls the space betwen rows
        counter = 1
        copy_target_vars = copy.deepcopy(target_vars)
        try:
            for target_var in copy_target_vars:
                df_target = df[df[dep]==target_var]
                ax1 = plt.subplot(rows,cols,counter)
                df_target.groupby(ts_column).mean().plot(subplots=False,ax=ax1)
                ax1.set_title('Time Series plots for '+dep + ' value = '+target_var)
                counter += 1
        except:
            plt.close('all')
            fig = plot_fast_average_num_by_cat(df_target, datevars, num_vars, verbose,kind="line")
        fig.suptitle('Time Series Plot by %s: Continuous Variables Pair' %ts_column, fontsize=15, y=1.01)
    if verbose == 2:
        imgdata_list.append(save_image_data(fig, chart_format,
                        plot_name, dep, mk_dir))
        image_count += 1
    return imgdata_list
    ############# End of Date vars plotting #########################

# This little function classifies columns into 4 types: categorical, continuous, boolean and
# certain columns that have only one value repeated that they are useless and must be removed from dataset
#Subtract RIGHT_LIST from LEFT_LIST to produce a new list
### This program is USED VERY HEAVILY so be careful about changing it
def list_difference(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst

######## Find ANY word in columns to identify ANY TYPE OF columns
####### search_for_list = ["Date","DATE", "date"], any words you want to search for it
####### columns__list and word refer to columns in the dataset that is the target dataset
####### Both columns_list and search_for_list must be lists - otherwise it won't work
def search_for_word_in_list(columns_list, search_for_list):
    columns_list = columns_list[:]
    search_for_list = search_for_list[:]
    lst=[]
    for src in search_for_list:
        for word in columns_list:
            result = re.findall (src, word)
            if len(result)>0:
                if word.endswith(src) and not word in lst:
                    lst.append(word)
            elif (word == 'id' or word == 'ID') and not word in lst:
                lst.append(word)
            else:
                continue
    return lst

### This is a small program to look for keywords such as "id" in a dataset to see if they are ID variables
### If that doesn't work, it then compares the len of the dataframe to the variable's unique values. If
###	they match, it means that the variable could be an ID variable. If not, it goes with the name of
###	of the ID variable through a keyword match with "id" or some such keyword in dataset's columns.
###  This is a small program to look for keywords such as "id" in a dataset to see if they are ID variables
###    If that doesn't work, it then compares the len of the dataframe to the variable's unique values. If
###     they match, it means that the variable could be an ID variable. If not, it goes with the name of
###     of the ID variable through a keyword match with "id" or some such keyword in dataset's columns.

def analyze_ID_columns(dfin,columns_list):
    columns_list = columns_list[:]
    dfin = dfin[:]
    IDcols_final = []
    IDcols = search_for_word_in_list(columns_list,
        ['ID','Identifier','NUMBER','No','Id','Num','num','_no','.no','Number','number','_id','.id'])
    if IDcols == []:
        for eachcol in columns_list:
            if len(dfin) == len(dfin[eachcol].unique()) and dfin[eachcol].dtype != float:
                IDcols_final.append(eachcol)
    else:
        for each_col in IDcols:
            if len(dfin) == len(dfin[each_col].unique()) and dfin[each_col].dtype != float:
                IDcols_final.append(each_col)
    if IDcols_final == [] and IDcols != []:
        IDcols_final = IDcols
    return IDcols_final

# THESE FUNCTIONS ASSUME A DIRTY DATASET" IN A PANDAS DATAFRAME AS Inum_j}lotsUT
# AND CONVERT THEM INTO A DATASET FIT FOR ANALYSIS IN THE END
# In [ ]:
# this function starts with dividing columns into 4 types: categorical, continuous, boolean and to_delete
# The To_Delete columns have only one unique value and can be removed from the dataset
def start_classifying_vars(dfin, verbose):
    dfin = dfin[:]
    cols_to_delete = []
    boolean_vars = []
    categorical_vars = []
    continuous_vars = []
    discrete_vars = []
    totrows = dfin.shape[0]
    if totrows == 0:
        print('Error: No rows in dataset. Check your input again...')
        return cols_to_delete, boolean_vars, categorical_vars, continuous_vars, discrete_vars, dfin
    for col in dfin.columns:
        if col == 'source':
            continue
        elif len(dfin[col].value_counts()) <= 1:
            cols_to_delete.append(dfin[col].name)
            print('    Column %s has only one value hence it will be dropped' %dfin[col].name)
        elif dfin[col].dtype==object:
            if (dfin[col].str.len()).any()>50:
                cols_to_delete.append(dfin[col].name)
                continue
            elif search_for_word_in_list([col],['DESCRIPTION','DESC','desc','Text','text']):
                cols_to_delete.append(dfin[col].name)
                continue
            elif len(dfin.groupby(col)) == 1:
                cols_to_delete.append(dfin[col].name)
                continue
            elif dfin[col].isnull().sum() > 0:
                missing_rows=dfin[col].isnull().sum()
                pct_missing = float(missing_rows)/float(totrows)
                if pct_missing > 0.90:
                    if verbose <= 1:
                        print('Pct of Missing Values in %s exceed 90 pct, hence will be dropped...' %col)
                    cols_to_delete.append(dfin[col].name)
                    continue
                elif len(dfin.groupby(col)) == 2:
                    boolean_vars.append(dfin[col].name)
                    py_version = sys.version_info[0]
                    if py_version < 3:
                        # This is the Python 2 Version
                        try:

                            item_mode = dfin[col].mode().mode[0]
                        except:
                            print('''Scipy.stats package not installed in your Python2. Get it installed''')
                    else:
                        # This is the Python 3 Version
                        try:

                            item_mode = dfin[col].mode()[0]
                        except:
                            print('''Statistics package not installed in your Python3. Get it installed''')
                    dfin[col].fillna(item_mode,inplace=True)
                    continue
                elif len(dfin.groupby(col)) < 20 and len(dfin.groupby(col)) > 1:
                    categorical_vars.append(dfin[col].name)
                    continue
                else:
                    discrete_vars.append(dfin[col].name)
                    continue
            elif len(dfin.groupby(col)) == 2:
                boolean_vars.append(dfin[col].name)
                continue
            elif len(dfin.groupby(col)) < 20 and len(dfin.groupby(col)) > 1:
                categorical_vars.append(dfin[col].name)
                continue
            else:
                discrete_vars.append(dfin[col].name)
        elif dfin[col].dtype=='int64' or dfin[col].dtype=='int32':
            if len(dfin[col].value_counts()) <= 15:
                categorical_vars.append(dfin[col].name)
        else:
            if dfin[col].isnull().sum() > 0:
                missing_rows=dfin[col].isnull().sum()
                pct_missing = float(missing_rows)/float(totrows)
                if pct_missing > 0.90:
                    if verbose <= 1:
                        print('Pct of Missing Values in %s exceed 90 pct, hence will be dropped...' %col)
                    cols_to_delete.append(dfin[col].name)
                    continue
                elif len(dfin.groupby(col)) == 2:
                    boolean_vars.append(dfin[col].name)
                    py_version = sys.version_info[0]
                    if py_version < 3:
                        # This is the Python 2 Version
                        try:

                            item_mode = dfin[col].mode().mode[0]
                        except:
                            print('''Scipy.stats package not installed in your Python2. Get it installed''')
                    else:
                        # This is the Python 3 Version
                        try:

                            item_mode = dfin[col].mode()[0]
                        except:
                            print('''Statistics package not installed in your Python3. Get it installed''')
                    dfin[col].fillna(item_mode,inplace=True)
                    continue
                else:
                    if len(dfin[col].value_counts()) <= 25 and len(dfin) >= 250:
                        categorical_vars.append(dfin[col].name)
                    else:
                        continuous_vars.append(dfin[col].name)
            elif len(dfin.groupby(col)) == 2:
                boolean_vars.append(dfin[col].name)
                continue
            else:
                if len(dfin[col].value_counts()) <= 25 and len(dfin) >= 250:
                    categorical_vars.append(dfin[col].name)
                else:
                    continuous_vars.append(dfin[col].name)
    return cols_to_delete, boolean_vars, categorical_vars, continuous_vars, discrete_vars, dfin

#### this is the MAIN ANALYSIS function that calls the start_classifying_vars and then
#### takes that result and divides categorical vars into 2 additional types: discrete vars and bool vars
def analyze_columns_in_dataset(dfx,IDcolse,verbose):
    dfx = dfx[:]
    IDcolse = IDcolse[:]
    cols_delete, bool_vars, cats, nums, discrete_string_vars, dft = start_classifying_vars(dfx,verbose)
    continuous_vars = nums
    if nums != []:
        for k in nums:
            if len(dft[k].unique())==2:
                bool_vars.append(k)
            elif len(dft[k].unique())<=20:
                cats.append(k)
            elif (np.array(dft[k]).dtype=='float64' or np.array(dft[k]).dtype=='int64') and (k not in continuous_vars):
                if len(dft[k].value_counts()) <= 25:
                    cats.append(k)
                else:
                    continuous_vars.append(k)
            elif dft[k].dtype==object:
                discrete_string_vars.append(k)
            elif k in continuous_vars:
                continue
            else:
                print('The %s variable could not be classified into any known type' % k)
    #print(cols_delete, bool_vars, cats, continuous_vars, discrete_string_vars)
    date_vars = search_for_word_in_list(dfx.columns.tolist(),['Date','DATE','date','TIME','time',
                                                   'Time','Year','Yr','year','yr','timestamp',
                                                   'TimeStamp','TIMESTAMP','Timestamp','Time Stamp'])
    date_vars = [x for x in date_vars if x not in find_remove_duplicates(cats+bool_vars) ]
    if date_vars == []:
        for col in continuous_vars:
            if dfx[col].dtype==int:
                if dfx[col].min() > 1900 or dfx[col].max() < 2100:
                    date_vars.append(col)
        for col in discrete_string_vars:
            try:
                dfx.index = pd.to_datetime(dfx.pop(col), infer_datetime_format=True)
            except:
                continue
    if isinstance(dfx.index, pd.DatetimeIndex):
        date_vars = [dfx.index.name]
    continuous_vars=list_difference(list_difference(continuous_vars,date_vars),IDcolse)
    #cats =  list_difference(continuous_vars, cats)
    cats=list_difference(cats,date_vars)
    discrete_string_vars=list_difference(list_difference(discrete_string_vars,date_vars),IDcolse)
    return cols_delete, bool_vars, cats, continuous_vars, discrete_string_vars,date_vars, dft

# Removes duplicates from a list to return unique values - USED ONLYONCE
def find_remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
#################################################################################
def load_file_dataframe(dataname, sep=",", header=0, verbose=0, nrows=None,parse_dates=False):
    start_time = time.time()
    ###########################  This is where we load file or data frame ###############
    if isinstance(dataname,str):
        #### this means they have given file name as a string to load the file #####
        codex_flag = False
        codex = ['ascii', 'utf-8', 'iso-8859-1', 'cp1252', 'latin1']
        if dataname != '' and dataname.endswith(('csv')):
            try:
                dfte = pd.read_csv(dataname, sep=sep, header=header, encoding=None,
                                parse_dates=parse_dates)
                if not nrows is None:
                    if nrows < dfte.shape[0]:
                        print('    max_rows_analyzed is smaller than dataset shape %d...' %dfte.shape[0])
                        dfte = dfte.sample(nrows, replace=False, random_state=99)
                        print('        randomly sampled %d rows from read CSV file' %nrows)
                print('Shape of your Data Set loaded: %s' %(dfte.shape,))
                if len(np.array(list(dfte))[dfte.columns.duplicated()]) > 0:
                    print('You have duplicate column names in your data set. Removing duplicate columns now...')
                    dfte = dfte[list(dfte.columns[~dfte.columns.duplicated(keep='first')])]
                return dfte
            except:
                codex_flag = True
        if codex_flag:
            for code in codex:
                try:
                    dfte = pd.read_csv(dataname, sep=sep, header=header, encoding=None, nrows=nrows,
                                    skiprows=skip_function, parse_dates=parse_dates)
                except:
                    print('    pandas %s encoder does not work for this file. Continuing...' %code)
                    continue
        elif dataname.endswith(('xlsx','xls','txt')):
            #### It's very important to get header rows in Excel since people put headers anywhere in Excel#
            if nrows is None:
                dfte = pd.read_excel(dataname,header=header, parse_dates=parse_dates)
            else:
                dfte = pd.read_excel(dataname,header=header, nrows=nrows, parse_dates=parse_dates)
            print('Shape of your Data Set loaded: %s' %(dfte.shape,))
            return dfte
        else:
            print('    Filename is an empty string or file not able to be loaded')
            return None
    elif isinstance(dataname,pd.DataFrame):
        #### this means they have given a dataframe name to use directly in processing #####
        if nrows is None:
            dfte = copy.deepcopy(dataname)
        else:
            if nrows < dataname.shape[0]:
                print('    Since nrows is smaller than dataset, loading random sample of %d rows into pandas...' %nrows)
                dfte = dataname.sample(n=nrows, replace=False, random_state=99)
            else:
                dfte = copy.deepcopy(dataname)
        print('Shape of your Data Set loaded: %s' %(dfte.shape,))
        if len(np.array(list(dfte))[dfte.columns.duplicated()]) > 0:
            print('You have duplicate column names in your data set. Removing duplicate columns now...')
            dfte = dfte[list(dfte.columns[~dfte.columns.duplicated(keep='first')])]
        return dfte
    else:
        print('Dataname input must be a filename with path to that file or a Dataframe')
        return None
##########################################################################################
import copy
def classify_print_vars(filename,sep, max_rows_analyzed, max_cols_analyzed,
                        depVar='',dfte=None, header=0,verbose=0):
    corr_limit = 0.7  ### This limit represents correlation above this, vars will be removed
    start_time=time.time()
    if filename:
        dataname = copy.deepcopy(filename)
        parse_dates = True
    else:
        dataname = copy.deepcopy(dfte)
        parse_dates = False
    dfte = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, 
                    nrows=max_rows_analyzed, parse_dates=parse_dates)
    orig_preds = [x for x in list(dfte) if x not in [depVar]]
    #################    CLASSIFY  COLUMNS   HERE    ######################
    var_df = classify_columns(dfte[orig_preds], verbose)
    #####       Classify Columns   ################
    IDcols = var_df['id_vars']
    discrete_string_vars = var_df['nlp_vars']+var_df['discrete_string_vars']
    cols_delete = var_df['cols_delete']
    bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
    int_vars = var_df['int_vars']
    categorical_vars = var_df['cat_vars'] + var_df['factor_vars'] + int_vars + bool_vars
    date_vars = var_df['date_vars']
    if len(var_df['continuous_vars'])==0 and len(int_vars)>0:
        continuous_vars = var_df['int_vars']
        categorical_vars = list_difference(categorical_vars, int_vars)
        int_vars = []
    else:
        continuous_vars = var_df['continuous_vars']
    #### from now you can use wordclouds on discrete_string_vars ######################
    preds = [x for x in orig_preds if x not in IDcols+cols_delete]
    if len(IDcols+cols_delete) == 0:
        print('        No variables removed since no ID or low-information variables found in data set')
    else:
        print('        %d variables removed since they were ID or low-information variables'
                                %len(IDcols+cols_delete))
        if verbose >= 1:
            print('    List of variables removed: %s' %(IDcols+cols_delete))
    #############    Sample data if too big and find problem type   #############################
    if dfte.shape[0]>= max_rows_analyzed:
        print('Since Number of Rows in data %d exceeds maximum, randomly sampling %d rows for EDA...' %(len(dfte),max_rows_analyzed))
        dft = dfte.sample(max_rows_analyzed, random_state=0)
    else:
        dft = copy.deepcopy(dfte)
    ###### This is where you find what type the dependent variable is ########
    if type(depVar) == str:
        if depVar == '':
            cols_list = list(dft)
            problem_type = 'Clustering'
            classes = []
        else:
            try:
                problem_type = analyze_problem_type(dft, depVar,verbose)
            except:
                print('Could not find given target var in data set. Please check input')
                ### return the data frame as is ############
                return dfte
            cols_list = list_difference(list(dft),depVar)
            if dft[depVar].dtype == object:
                classes = dft[depVar].factorize()[1].tolist()
                #### You dont have to convert it since most charts can take string vars as target ####
                #dft[depVar] = dft[depVar].factorize()[0]
            elif dft[depVar].dtype == np.int64:
                classes = dft[depVar].factorize()[1].tolist()
            elif dft[depVar].dtype == bool:
                dft[depVar] = dft[depVar].astype(int)
                classes =  dft[depVar].unique().astype(int).tolist()
            elif dft[depVar].dtype == float and problem_type.endswith('Classification'):
                classes = dft[depVar].factorize()[1].tolist()
            else:
                classes = []
    elif depVar == None:
            cols_list = list(dft)
            problem_type = 'Clustering'
            classes = []
    else:
        depVar1 = depVar[0]
        problem_type = analyze_problem_type(dft, depVar1)
        cols_list = list_difference(list(dft), depVar1)
        if dft[depVar1].dtype == object:
            classes = dft[depVar1].factorize()[1].tolist()
            #### You dont have to convert it since most charts can take string vars as target ####
            #dft[depVar] = dft[depVar].factorize()[0]
        elif dft[depVar1].dtype == np.int64:
            classes = dft[depVar1].factorize()[1].tolist()
        elif dft[depVar1].dtype == bool:
            classes =  dft[depVar].unique().astype(int).tolist()
        elif dft[depVar1].dtype == float and problem_type.endswith('Classification'):
            classes = dft[depVar1].factorize()[1].tolist()
        else:
            classes = []
        print('Since AutoViz cannot visualize multiple targets, selecting the first one from list: %s' %depVar1)
        depVar = copy.deepcopy(depVar1)
    #############  Check if there are too many columns to visualize  ################
    if len(preds) >= max_cols_analyzed:
        #########     In that case, SELECT IMPORTANT FEATURES HERE   ######################
        if problem_type.endswith('Classification') or problem_type == 'Regression':
            print('Number of variables = %d exceeds limit, finding top %d variables through XGBoost' %(len(
                                            preds), max_cols_analyzed))
            important_features,num_vars, _ = find_top_features_xgb(dft,preds,continuous_vars,
                                                         depVar,problem_type,corr_limit,verbose)
            if len(important_features) >= max_cols_analyzed:
                print('    Since number of features selected is greater than max columns analyzed, limiting to %d variables' %max_cols_analyzed)
                important_features = important_features[:max_cols_analyzed]
            dft = dft[important_features+[depVar]]
            #### Time to  classify the important columns again ###
            var_df = classify_columns(dft[important_features], verbose)
            IDcols = var_df['id_vars']
            discrete_string_vars = var_df['nlp_vars']+var_df['discrete_string_vars']
            cols_delete = var_df['cols_delete']
            bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
            int_vars = var_df['int_vars']
            categorical_vars = var_df['cat_vars'] + var_df['factor_vars'] + int_vars + bool_vars
            if len(var_df['continuous_vars'])==0 and len(int_vars)>0:
                continuous_vars = var_df['int_vars']
                categorical_vars = list_difference(categorical_vars, int_vars)
                int_vars = []
            else:
                continuous_vars = var_df['continuous_vars']
            date_vars = var_df['date_vars']
            preds = [x for x in important_features if x not in IDcols+cols_delete+discrete_string_vars]
            if len(IDcols+cols_delete+discrete_string_vars) == 0:
                print('    No variables removed since no ID or low-information variables found in data')
            else:
                print('    %d variables removed since they were ID or low-information variables'
                                        %len(IDcols+cols_delete+discrete_string_vars))
            if verbose >= 1:
                print('    List of variables removed: %s' %(IDcols+cols_delete+discrete_string_vars))
            dft = dft[preds+[depVar]]
        else:
            continuous_vars = continuous_vars[:max_cols_analyzed]
            print('%d numeric variables in data exceeds limit, taking top %d variables' %(len(
                                            continuous_vars), max_cols_analyzed))
            if verbose >= 1:
                print('    List of variables selected: %s' %(continuous_vars[:max_cols_analyzed]))
    elif len(continuous_vars) < 1:
        print('No continuous variables in this data set. No visualization can be performed')
        ### Return data frame as is #####
        return dfte
    else:
        #########     If above 1 but below limit, leave features as it is   ######################
        if not isinstance(depVar, list):
            if depVar != '':
                dft = dft[preds+[depVar]]
        else:
            dft = dft[preds+depVar]
    ###################   Time to reduce cat vars which have more than 30 categories #############
    #discrete_string_vars += np.array(categorical_vars)[dft[categorical_vars].nunique()>30].tolist()
    #categorical_vars = left_subtract(categorical_vars,np.array(
    #    categorical_vars)[dft[categorical_vars].nunique()>30].tolist())
    #############   Next you can print them if verbose is set to print #########
    ppt = pprint.PrettyPrinter(indent=4)
    if verbose==1 and len(cols_list) <= max_cols_analyzed:
        marthas_columns(dft,verbose)
        print("   Columns to delete:")
        ppt.pprint('   %s' %cols_delete)
        print("   Boolean variables %s ")
        ppt.pprint('   %s' %bool_vars)
        print("   Categorical variables %s ")
        ppt.pprint('   %s' %categorical_vars)
        print("   Continuous variables %s " )
        ppt.pprint('   %s' %continuous_vars)
        print("   Discrete string variables %s " )
        ppt.pprint('   %s' %discrete_string_vars)
        print("   Date and time variables %s " )
        ppt.pprint('   %s' %date_vars)
        print("   ID variables %s ")
        ppt.pprint('   %s' %IDcols)
        print("   Target variable %s ")
        ppt.pprint('   %s' %depVar)
    elif verbose==1 and len(cols_list) > 30:
        print('   Total columns > 30, too numerous to print.')
    return dft,depVar,IDcols,bool_vars,categorical_vars,continuous_vars,discrete_string_vars,date_vars,classes,problem_type, cols_list
####################################################################
def marthas_columns(data,verbose=0):
    """
    This program is named  in honor of my one of students who came up with the idea for it.
    It's a neat way of printing data types and information compared to the boring describe() function in Pandas.
    """
    data = data[:]
    print('Data Set Shape: %d rows, %d cols' % data.shape)
    if data.shape[1] > 30:
        print('Too many columns to print')
    else:
        if verbose==1:
            print('Data Set columns info:')
            for col in data.columns:
                print('* %s: %d nulls, %d unique vals, most common: %s' % (
                        col,
                        data[col].isnull().sum(),
                        data[col].nunique(),
                        data[col].value_counts().head(2).to_dict()
                    ))
            print('--------------------------------------------------------------------')
################################################
################################################################################
######### NEW And FAST WAY to CLASSIFY COLUMNS IN A DATA SET #######
################################################################################
def classify_columns(df_preds, verbose=0):
    """
    This actually does Exploratory data analysis - it means this function performs EDA
    ######################################################################################
    Takes a dataframe containing only predictors to be classified into various types.
    DO NOT SEND IN A TARGET COLUMN since it will try to include that into various columns.
    Returns a data frame containing columns and the class it belongs to such as numeric,
    categorical, date or id column, boolean, nlp, discrete_string and cols to delete...
    ####### Returns a dictionary with 10 kinds of vars like the following: # continuous_vars,int_vars
    # cat_vars,factor_vars, bool_vars,discrete_string_vars,nlp_vars,date_vars,id_vars,cols_delete
    """
    train = copy.deepcopy(df_preds)
    #### If there are 30 chars are more in a discrete_string_var, it is then considered an NLP variable
    max_nlp_char_size = 30
    max_cols_to_print = 30
    print('############## C L A S S I F Y I N G  V A R I A B L E S  ####################')
    print('Classifying variables in data set...')
    #### Cat_Limit defines the max number of categories a column can have to be called a categorical colum
    cat_limit = 35
    float_limit = 15 #### Make this limit low so that float variables below this limit become cat vars ###
    def add(a,b):
        return a+b
    sum_all_cols = dict()
    orig_cols_total = train.shape[1]
    #Types of columns
    cols_delete = [col for col in list(train) if (len(train[col].value_counts()) == 1
                                       ) | (train[col].isnull().sum()/len(train) >= 0.90)]
    train = train[left_subtract(list(train),cols_delete)]
    var_df = pd.Series(dict(train.dtypes)).reset_index(drop=False).rename(
                        columns={0:'type_of_column'})
    sum_all_cols['cols_delete'] = cols_delete
    var_df['bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['bool','object']
                        and len(train[x['index']].value_counts()) == 2 else 0, axis=1)
    string_bool_vars = list(var_df[(var_df['bool'] ==1)]['index'])
    sum_all_cols['string_bool_vars'] = string_bool_vars
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                            np.uint16, np.uint32, np.uint64,
                            'int8','int16','int32','int64',
                            'float16','float32','float64'] and len(
                        train[x['index']].value_counts()) == 2 else 0, axis=1)
    num_bool_vars = list(var_df[(var_df['num_bool'] ==1)]['index'])
    sum_all_cols['num_bool_vars'] = num_bool_vars
    ######   This is where we take all Object vars and split them into diff kinds ###
    discrete_or_nlp = var_df.apply(lambda x: 1 if x['type_of_column'] in ['object']  and x[
        'index'] not in string_bool_vars+cols_delete else 0,axis=1)
    ######### This is where we figure out whether a string var is nlp or discrete_string var ###
    var_df['nlp_strings'] = 0
    var_df['discrete_strings'] = 0
    var_df['cat'] = 0
    var_df['id_col'] = 0
    discrete_or_nlp_vars = var_df.loc[discrete_or_nlp==1]['index'].values.tolist()
    copy_discrete_or_nlp_vars = copy.deepcopy(discrete_or_nlp_vars)
    if len(discrete_or_nlp_vars) > 0:
        for col in copy_discrete_or_nlp_vars:
            #### first fill empty or missing vals since it will blowup ###
            train[col] = train[col].fillna('emptyspace')
            if train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= 50 and len(train[col].value_counts()
                        ) >= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= max_nlp_char_size and train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) < 50 and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) == len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                var_df.loc[var_df['index']==col,'cat'] = 1
    nlp_vars = list(var_df[(var_df['nlp_strings'] ==1)]['index'])
    sum_all_cols['nlp_vars'] = nlp_vars
    discrete_string_vars = list(var_df[(var_df['discrete_strings'] ==1) ]['index'])
    sum_all_cols['discrete_string_vars'] = discrete_string_vars
    ###### This happens only if a string column happens to be an ID column #######
    #### DO NOT Add this to ID_VARS yet. It will be done later.. Dont change it easily...
    #### Category DTYPE vars are very special = they can be left as is and not disturbed in Python. ###
    var_df['dcat'] = var_df.apply(lambda x: 1 if str(x['type_of_column'])=='category' else 0,
                            axis=1)
    factor_vars = list(var_df[(var_df['dcat'] ==1)]['index'])
    sum_all_cols['factor_vars'] = factor_vars
    ########################################################################
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                         np.uint16, np.uint32, np.uint64,
                         'int8','int16',
                        'int32','int64']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ######### This is where we figure out whether a numeric col is date or id variable ###
    var_df['int'] = 0
    var_df['date_time'] = 0
    ### if a particular column is date-time type, now set it as a date time variable ##
    var_df['date_time'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['<M8[ns]','datetime64[ns]']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ### this is where we save them as date time variables ###
    if len(var_df.loc[date_or_id==1]) != 0:
        for col in var_df.loc[date_or_id==1]['index'].values.tolist():
            if len(train[col].value_counts()) == len(train):
                if train[col].min() < 1900 or train[col].max() > 2050:
                    var_df.loc[var_df['index']==col,'id_col'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                if train[col].min() < 1900 or train[col].max() > 2050:
                    if col not in num_bool_vars:
                        var_df.loc[var_df['index']==col,'int'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        if col not in num_bool_vars:
                            var_df.loc[var_df['index']==col,'int'] = 1
    else:
        pass
    int_vars = list(var_df[(var_df['int'] ==1)]['index'])
    date_vars = list(var_df[(var_df['date_time'] == 1)]['index'])
    id_vars = list(var_df[(var_df['id_col'] == 1)]['index'])
    sum_all_cols['int_vars'] = int_vars
    copy_date_vars = copy.deepcopy(date_vars)
    for date_var in copy_date_vars:
        #### This test is to make sure sure date vars are actually date vars
        try:
            pd.to_datetime(train[date_var],infer_datetime_format=True)
        except:
            ##### if not a date var, then just add it to delete it from processing
            cols_delete.append(date_var)
            date_vars.remove(date_var)
    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    sum_all_cols['cols_delete'] = cols_delete
    ## This is an EXTREMELY complicated logic for cat vars. Don't change it unless you test it many times!
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= float_limit and len(train[col].value_counts()) <= len(train):
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if col not in num_bool_vars:
                    var_df.loc[var_df['index']==col,'numeric'] = 1
    cat_vars = list(var_df[(var_df['cat'] ==1)]['index'])
    continuous_vars = list(var_df[(var_df['numeric'] ==1)]['index'])
    ########  V E R Y    I M P O R T A N T   ###################################################
    ##### There are a couple of extra tests you need to do to remove abberations in cat_vars ###
    cat_vars_copy = copy.deepcopy(cat_vars)
    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['continuous_vars'] = continuous_vars
    sum_all_cols['id_vars'] = id_vars
    ###### This is where you consoldate the numbers ###########
    var_dict_sum = dict(zip(var_df.values[:,0], var_df.values[:,2:].sum(1)))
    for col, sumval in var_dict_sum.items():
        if sumval == 0:
            print('%s of type=%s is not classified' %(col,train[col].dtype))
        elif sumval > 1:
            print('%s of type=%s is classified into more then one type' %(col,train[col].dtype))
        else:
            pass
    ###############  This is where you print all the types of variables ##############
    ####### Returns 8 vars in the following order: continuous_vars,int_vars,cat_vars,
    ###  string_bool_vars,discrete_string_vars,nlp_vars,date_or_id_vars,cols_delete
    if verbose == 1:
        print("    Number of Numeric Columns = ", len(continuous_vars))
        print("    Number of Integer-Categorical Columns = ", len(int_vars))
        print("    Number of String-Categorical Columns = ", len(cat_vars))
        print("    Number of Factor-Categorical Columns = ", len(factor_vars))
        print("    Number of String-Boolean Columns = ", len(string_bool_vars))
        print("    Number of Numeric-Boolean Columns = ", len(num_bool_vars))
        print("    Number of Discrete String Columns = ", len(discrete_string_vars))
        print("    Number of NLP String Columns = ", len(nlp_vars))
        print("    Number of Date Time Columns = ", len(date_vars))
        print("    Number of ID Columns = ", len(id_vars))
        print("    Number of Columns to Delete = ", len(cols_delete))
    if verbose == 2:
        marthas_columns(df_preds,verbose=1)
        print("    Numeric Columns: %s" %continuous_vars[:max_cols_to_print])
        print("    Integer-Categorical Columns: %s" %int_vars[:max_cols_to_print])
        print("    String-Categorical Columns: %s" %cat_vars[:max_cols_to_print])
        print("    Factor-Categorical Columns: %s" %factor_vars[:max_cols_to_print])
        print("    String-Boolean Columns: %s" %string_bool_vars[:max_cols_to_print])
        print("    Numeric-Boolean Columns: %s" %num_bool_vars[:max_cols_to_print])
        print("    Discrete String Columns: %s" %discrete_string_vars[:max_cols_to_print])
        print("    NLP text Columns: %s" %nlp_vars[:max_cols_to_print])
        print("    Date Time Columns: %s" %date_vars[:max_cols_to_print])
        print("    ID Columns: %s" %id_vars[:max_cols_to_print])
        print("    Columns that will not be considered in modeling: %s" %cols_delete[:max_cols_to_print])
    ##### now collect all the column types and column names into a single dictionary to return!
    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()])
    if len_sum_all_cols == orig_cols_total:
        print('    %d Predictors classified...' %orig_cols_total)
        #print('        This does not include the Target column(s)')
    else:
        print('No of columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]
        if len(left_subtract(list(train),flat_list)) == 0:
            print(' Missing columns = None')
        else:
            print(' Missing columns = %s' %left_subtract(list(train),flat_list))
    return sum_all_cols
#################################################################################
from collections import Counter
import time
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
################################################################################
from collections import defaultdict
from collections import OrderedDict
import time
def return_dictionary_list(lst_of_tuples):
    """ Returns a dictionary of lists if you send in a list of Tuples"""
    orDict = defaultdict(list)
    # iterating over list of tuples
    for key, val in lst_of_tuples:
        orDict[key].append(val)
    return orDict
##################################################################################
def remove_variables_using_fast_correlation(df, numvars, modeltype, target,
                                corr_limit = 0.70,verbose=0):
    """
    #### THIS METHOD IS KNOWN AS SULO METHOD in HONOR OF my mother SULOCHANA SESHADRI #######
    This highly efficient method removes variables that are highly correlated using a series of
    pair-wise correlation knockout rounds. It is extremely fast and hence can work on thousands
    of variables in less than a minute, even on a laptop. You need to send in a list of numeric
    variables and that's all! The method defines high Correlation as anything over 0.70 (absolute)
    but this can be changed. If two variables have absolute correlation higher than this, they
    will be marked, and using a process of elimination, one of them will get knocked out:
    To decide order of variables to keep, we use mutuail information score to select. MIS returns
    a ranked list of these correlated variables: when we select one, we knock out others
    that it is correlated to. Then we select next var. This way we knock out correlated variables.
    Finally we are left with uncorrelated variables that are also highly important (mutual score).
    ##############  YOU MUST INCLUDE THE ABOVE MESSAGE IF YOU COPY THIS CODE IN YOUR LIBRARY #####
    """
    print('    Removing correlated variables from %d numerics using SULO method' %len(numvars))
    correlation_dataframe = df[numvars].corr()
    a = correlation_dataframe.values
    col_index = correlation_dataframe.columns.tolist()
    index_triupper = list(zip(np.triu_indices_from(a,k=1)[0],np.triu_indices_from(a,k=1)[1]))
    high_corr_index_list = [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])>=corr_limit)]
    low_corr_index_list =  [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])<corr_limit)]
    tuple_list = [y for y in [index_triupper[x[0]] for x in high_corr_index_list]]
    correlated_pair = [(col_index[tuple[0]],col_index[tuple[1]]) for tuple in tuple_list]
    corr_pair_dict = dict(return_dictionary_list(correlated_pair))
    keys_in_dict = list(corr_pair_dict.keys())
    reverse_correlated_pair = [(y,x) for (x,y) in correlated_pair]
    reverse_corr_pair_dict = dict(return_dictionary_list(reverse_correlated_pair))
    for key, val in reverse_corr_pair_dict.items():
        if key in keys_in_dict:
            if len(key) > 1:
                corr_pair_dict[key] += val
        else:
            corr_pair_dict[key] = val
    flat_corr_pair_list = [item for sublist in correlated_pair for item in sublist]
    #### You can make it a dictionary or a tuple of lists. We have chosen the latter here to keep order intact.
    corr_pair_count_dict = count_freq_in_list(flat_corr_pair_list)
    corr_list = [k for (k,v) in corr_pair_count_dict]
    ###### This is for ordering the variables in the highest to lowest importance to target ###
    if len(corr_list) == 0:
        final_list = list(correlation_dataframe)
        print('Selecting all (%d) variables since none of them are highly correlated...' %len(numvars))
        return numvars
    else:
        max_feats = len(corr_list)
        if modeltype == 'Regression':
            sel_function = mutual_info_regression
            fs = SelectKBest(score_func=sel_function, k=max_feats)
            fs.fit(df[corr_list], df[target])
            mutual_info = dict(zip(corr_list,fs.scores_))
        else:
            sel_function = mutual_info_classif
            fs = SelectKBest(score_func=sel_function, k=max_feats)
            fs.fit(df[corr_list], df[target])
            mutual_info = dict(zip(corr_list,fs.scores_))
        #### The first variable in list has the highest correlation to the target variable ###
        sorted_by_mutual_info =[key for (key,val) in sorted(mutual_info.items(), key=lambda kv: kv[1],reverse=True)]
        #####   Now we select the final list of correlated variables ###########
        selected_corr_list = []
        #### select each variable by the highest mutual info and see what vars are correlated to it
        for each_corr_name in sorted_by_mutual_info:
            ### add the selected var to the selected_corr_list
            selected_corr_list.append(each_corr_name)
            for each_remove in corr_pair_dict[each_corr_name]:
                #### Now remove each variable that is highly correlated to the selected variable
                if each_remove in sorted_by_mutual_info:
                    sorted_by_mutual_info.remove(each_remove)
        ##### Now we combine the uncorrelated list to the selected correlated list above
        rem_col_list = left_subtract(list(correlation_dataframe),corr_list)
        final_list = rem_col_list + selected_corr_list
        if verbose >= 1:
            print('\nAfter removing highly correlated variables, following %d numeric vars selected: %s' %(len(final_list),final_list))
        return final_list
###############################################################################################
def count_freq_in_list(lst):
    """
    This counts the frequency of items in a list but MAINTAINS the order of appearance of items.
    This order is very important when you are doing certain functions. Hence this function!
    """
    temp=np.unique(lst)
    result = []
    for i in temp:
        result.append((i,lst.count(i)))
    return result

def find_corr_vars(correlation_dataframe,corr_limit = 0.70):
    """
    This returns a dictionary of counts of each variable and how many vars it is correlated to in the dataframe
    """
    flatten = lambda l: [item for sublist in l for item in sublist]
    flatten_items = lambda dic: [x for x in dic.items()]
    a = correlation_dataframe.values
    col_index = correlation_dataframe.columns.tolist()
    index_triupper = list(zip(np.triu_indices_from(a,k=1)[0],np.triu_indices_from(a,k=1)[1]))
    high_corr_index_list = [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])>=corr_limit)]
    low_corr_index_list =  [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])<corr_limit)]
    tuple_list = [y for y in [index_triupper[x[0]] for x in high_corr_index_list]]
    correlated_pair = [(col_index[tuple[0]],col_index[tuple[1]]) for tuple in tuple_list]
    correlated_pair_dict = dict(correlated_pair)
    flat_corr_pair_list = [item for sublist in correlated_pair for item in sublist]
    #### You can make it a dictionary or a tuple of lists. We have chosen the latter here to keep order intact.
    #corr_pair_count_dict = Counter(flat_corr_pair_list)
    corr_pair_count_dict = count_freq_in_list(flat_corr_pair_list)
    corr_list = list(set(flatten(flatten_items(correlated_pair_dict))))
    rem_col_list = left_subtract(list(correlation_dataframe),list(OrderedDict.fromkeys(flat_corr_pair_list)))
    return corr_pair_count_dict, rem_col_list, corr_list, correlated_pair_dict
#################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#######
def convert_train_test_cat_col_to_numeric(start_train, start_test, col):
    """
    ####  This is the easiest way to label encode object variables in both train and test
    #### This takes care of some categories that are present in train and not in test
    ###     and vice versa
    """
    start_train = copy.deepcopy(start_train)
    start_test = copy.deepcopy(start_test)
    if start_train[col].isnull().sum() > 0:
        start_train[col] = start_train[col].fillna("NA")
    train_categs = list(pd.unique(start_train[col].values))
    if not isinstance(start_test,str) :
        test_categs = list(pd.unique(start_test[col].values))
        categs_all = train_categs+test_categs
        dict_all =  return_factorized_dict(categs_all)
    else:
        dict_all = return_factorized_dict(train_categs)
    start_train[col] = start_train[col].map(dict_all)
    if not isinstance(start_test,str) :
        if start_test[col].isnull().sum() > 0:
            start_test[col] = start_test[col].fillna("NA")
        start_test[col] = start_test[col].map(dict_all)
    return start_train, start_test
###############################################################################
def return_factorized_dict(ls):
    """
    ######  Factorize any list of values in a data frame using this neat function
    if your data has any NaN's it automatically marks it as -1 and returns that for NaN's
    Returns a dictionary mapping previous values with new values.
    """
    factos = pd.unique(pd.factorize(ls)[0])
    categs = pd.unique(pd.factorize(ls)[1])
    if -1 in factos:
        categs = np.insert(categs,np.where(factos==-1)[0][0],np.nan)
    return dict(zip(categs,factos))
#################################################################################
################      Find top features using XGB     ###################
#################################################################################
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import copy
from sklearn.multiclass import OneVsRestClassifier
from collections import OrderedDict
################################################################################
###########################################################################################
############## CONVERSION OF STRING COLUMNS TO NUMERIC WITHOUT LABEL ENCODER #########
#######################################################################################
import copy
import pdb
def convert_a_column_to_numeric(x, col_dict=""):
    '''Function converts any pandas series (or column) consisting of string chars,
       into numeric values. It converts an all-string column to an all-number column.
       This is an amazing function which performs exactly like a Label Encoding
       except that it is simpler and faster'''
    if isinstance(col_dict, str):
        values = np.unique(x)
        values2nums = dict(zip(values,range(len(values))))
        convert_dict = dict(zip(range(len(values)),values))
        return x.replace(values2nums), convert_dict
    else:
        convert_dict = copy.deepcopy(col_dict)
        keys  = col_dict.keys()
        newkeys = np.unique(x)
        rem_keys = left_subtract(newkeys, keys)
        max_val = max(col_dict.values()) + 1
        for eachkey in rem_keys:
            convert_dict.update({eachkey:max_val})
            max_val += 1
        return x.replace(convert_dict)
#######################################################################################
def convert_a_mixed_object_column_to_numeric(x, col_dict=''):
    """
    This is the main utility that converts any string column to numeric.
    It does not need Label Encoder since it picks up an string that may not be in test data.
    """
    x = x.astype(str)
    if isinstance(col_dict, str):
        x, convert_dict = convert_a_column_to_numeric(x)
        convert_dict = dict([(v,k) for (k,v) in convert_dict.items()])
        return x, convert_dict
    else:
        x = convert_a_column_to_numeric(x, col_dict)
    return x
######################################################################################
def convert_all_object_columns_to_numeric(train, test):
    """
    #######################################################################################
    This is a utility that converts string columns to numeric WITHOUT LABEL ENCODER.
    The beauty of this utility is that it does not blow up when it finds strings in test not in train.
    #######################################################################################
    """
    train = copy.deepcopy(train)
    if train.dtypes.any()==object:
        lis=[]
        for row,column in train.dtypes.iteritems():
            if column == object:
                lis.append(row)
        #print('%d string variables identified' %len(lis))
        for everycol in lis:
            #print('    Converting %s to numeric' %everycol)
            try:
                train[everycol], train_dict = convert_a_mixed_object_column_to_numeric(train[everycol])
                if not isinstance(test, str):
                    test[everycol] = convert_a_mixed_object_column_to_numeric(test[everycol], train_dict)
            except:
                #print('Error converting %s column from string to numeric. Continuing...' %everycol)
                continue
    return train, test
###################################################################################
################      Find top features using XGB     ###################
################################################################################
from xgboost import XGBClassifier, XGBRegressor
def find_top_features_xgb(train,preds,numvars,target,modeltype,corr_limit=0.7,verbose=0):
    """
    This is a fast utility that uses XGB to find top features. You
    It returns a list of important features.
    Since it is XGB, you dont have to restrict the input to just numeric vars.
    You can send in all kinds of vars and it will take care of transforming it. Sweet!
    """
    train = copy.deepcopy(train)
    preds = copy.deepcopy(preds)
    numvars = copy.deepcopy(numvars)
    ######################   I M P O R T A N T ##############################################
    ###### This top_num decides how many top_n features XGB selects in each iteration.
    ####  There a total of 5 iterations. Hence 5x10 means maximum 50 features will be selected.
    #####  If there are more than 50 variables, then maximum 25% of its variables will be selected
    if len(preds) <= 50:
        top_num = 10
    else:
        ### the maximum number of variables will 25% of preds which means we divide by 5 and get 5% here
        ### The five iterations result in 5% being chosen in each iteration. Hence max 25% of variables!
        top_num = int(len(preds)*0.05)
    ######################   I M P O R T A N T ##############################################
    #### If there are more than 30 categorical variables in a data set, it is worth reducing features.
    ####  Otherwise. XGBoost is pretty good at finding the best features whether cat or numeric !
    n_splits = 5
    max_depth = 8
    max_cats = 5
    ######################   I M P O R T A N T ##############################################
    subsample =  0.7
    col_sub_sample = 0.7
    train = copy.deepcopy(train)
    start_time = time.time()
    test_size = 0.2
    seed = 1
    early_stopping = 5
    ####### All the default parameters are set up now #########
    kf = KFold(n_splits=n_splits)
    rem_vars = left_subtract(preds,numvars)
    catvars = copy.deepcopy(rem_vars)
    ############   I  M P O R T A N T ! I M P O R T A N T ! ######################
    ##### Removing the Cat Vars selection using Linear Methods since they fail so often.
    ##### Linear methods such as Chi2 or Mutual Information Score are not great
    ####  for feature selection since they can't handle large data and provide
    ####  misleading results for large data sets. Hence I am using XGBoost alone.
    ####  Also, another method of using Spearman Correlation for CatVars with 100's
    ####  of variables is very slow. Also, is not very clear is effective: only 3-4 vars
    ####   are removed. Hence for now, I am not going to use Spearman method. Perhaps later.
    ##############################################################################
    #if len(catvars) > max_cats:
    #    start_time = time.time()
    #    important_cats = remove_variables_using_fast_correlation(train,catvars,'spearman',
    #                         corr_limit,verbose)
    #    if verbose >= 1:
    #        print('Time taken for reducing highly correlated Categorical vars was %0.0f seconds' %(time.time()-start_time))
    #else:
    important_cats = copy.deepcopy(catvars)
    print('    No categorical feature reduction done. All %d Categorical vars selected ' %(len(catvars)))
    ########    Drop Missing value rows since XGB for some reason  #########
    ########    can't handle missing values in early stopping rounds #######
    train.dropna(axis=0,subset=preds+[target],inplace=True)
    if len(numvars) > 1:
        final_list = remove_variables_using_fast_correlation(train,numvars,modeltype,target,
                             corr_limit,verbose)
    else:
        final_list = copy.deepcopy(numvars)
    print('    Adding %s categorical variables to reduced numeric variables  of %d' %(
                            len(important_cats),len(final_list)))
    if  isinstance(final_list,np.ndarray):
        final_list = final_list.tolist()
    preds = final_list+important_cats
    #######You must convert category variables into integers ###############
    if len(important_cats) > 0:
        train, _ = convert_all_object_columns_to_numeric(train, "")
    ########   Dont move this train and y definition anywhere else ########
    y = train[target]
    print('############## F E A T U R E   S E L E C T I O N  ####################')
    important_features = []
    if modeltype == 'Regression':
        objective = 'reg:squarederror'
        model_xgb = XGBRegressor( n_estimators=100,subsample=subsample,objective=objective,
                                colsample_bytree=col_sub_sample,reg_alpha=0.5, reg_lambda=0.5,
                                 seed=1,n_jobs=-1,random_state=1)
        eval_metric = 'rmse'
    else:
        #### This is for Classifiers only
        classes = np.unique(train[target].values)
        if len(classes) == 2:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                colsample_bytree=col_sub_sample,gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=100,
                n_jobs=-1, nthread=None, objective='binary:logistic',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5,
                seed=1)
            eval_metric = 'logloss'
        else:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                        colsample_bytree=col_sub_sample, gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=100,
                n_jobs=-1, nthread=None, objective='multi:softmax',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5,
                seed=1)
            eval_metric = 'mlogloss'
    ####   This is where you start to Iterate on Finding Important Features ################
    save_xgb = copy.deepcopy(model_xgb)
    train_p = train[preds]
    if train_p.shape[1] < 10:
        iter_limit = 2
    else:
        iter_limit = int(train_p.shape[1]/5+0.5)
    print('Current number of predictors = %d ' %(train_p.shape[1],))
    print('    Finding Important Features using Boosted Trees algorithm...')
    try:
        for i in range(0,train_p.shape[1],iter_limit):
            new_xgb = copy.deepcopy(save_xgb)
            print('        using %d variables...' %(train_p.shape[1]-i))
            if train_p.shape[1]-i < iter_limit:
                X = train_p.iloc[:,i:]
                if modeltype == 'Regression':
                    train_part = int((1-test_size)*X.shape[0])
                    X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
                else:
                    X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                                test_size=test_size, random_state=seed)
                try:
                    eval_set = [(X_train,y_train),(X_cv,y_cv)]
                    model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,eval_set=eval_set,
                                        eval_metric=eval_metric,verbose=False)
                    important_features += pd.Series(model_xgb.get_booster().get_score(
                                importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                except:
                    new_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,eval_set=eval_set,
                                        eval_metric=eval_metric,verbose=False)
                    print('XGB has a bug in version xgboost 1.02 for feature importances. Try to install version 0.90 or 1.10 - continuing...')
                    important_features += pd.Series(new_xgb.get_booster().get_score(
                                importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                important_features = list(OrderedDict.fromkeys(important_features))
            else:
                X = train_p[list(train_p.columns.values)[i:train_p.shape[1]]]
                #### Split here into train and test #####
                if modeltype == 'Regression':
                    train_part = int((1-test_size)*X.shape[0])
                    X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
                else:
                    X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                                test_size=test_size, random_state=seed)
                eval_set = [(X_train,y_train),(X_cv,y_cv)]
                try:
                    model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,
                                  eval_set=eval_set,eval_metric=eval_metric,verbose=False)
                    important_features += pd.Series(model_xgb.get_booster().get_score(
                                importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                except:
                    new_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,
                                  eval_set=eval_set,eval_metric=eval_metric,verbose=False)
                    important_features += pd.Series(model_xgb.get_booster().get_score(
                                importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                important_features = list(OrderedDict.fromkeys(important_features))
    except:
        print('Finding top features using XGB is crashing. Continuing with all predictors...')
        important_features = copy.deepcopy(preds)
        return important_features, [], []
    important_features = list(OrderedDict.fromkeys(important_features))
    print('Found %d important features' %len(important_features))
    #print('    Time taken (in seconds) = %0.0f' %(time.time()-start_time))
    numvars = [x for x in numvars if x in important_features]
    important_cats = [x for x in important_cats if x in important_features]
    return important_features, numvars, important_cats
######################################################################################
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def return_stop_words():
    STOP_WORDS = ['it', "this", "that", "to", 'its', 'am', 'is', 'are', 'was', 'were', 'a',
                'an', 'the', 'and', 'or', 'of', 'at', 'by', 'for', 'with', 'about', 'between',
                 'into','above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'over',
                  'under', 'again', 'further', 'then', 'once', 'all', 'any', 'both', 'each',
                   'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
                    'than', 'too', 'very', 's', 't', 'can', 'just', 'd', 'll', 'm', 'o', 're',
                    've', 'y', 'ain', 'ma']
    add_words = ["s", "m",'you', 'not',  'get', 'no', 'via', 'one', 'still', 'us', 'u','hey','hi','oh','jeez',
                'the', 'a', 'in', 'to', 'of', 'i', 'and', 'is', 'for', 'on', 'it', 'got','aww','awww',
                'not', 'my', 'that', 'by', 'with', 'are', 'at', 'this', 'from', 'be', 'have', 'was',
                '', ' ', 'say', 's', 'u', 'ap', 'afp', '...', 'n', '\\']
    stop_words = list(set(STOP_WORDS+add_words))
    return sorted(stop_words)

def draw_wordcloud_from_dataframe(dataframe, column, chart_format, plot_name, depVar, mk_dir, verbose=0):
    """
    This handy function draws a dataframe column using Wordcloud library and nltk.
    """
    imgdata_list = []
    replace_spaces = re.compile('[/(){}\[\]\|@,;]')
    remove_special_chars = re.compile('[^0-9a-z #+_]')
    STOPWORDS = return_stop_words()
    remove_ip_addr = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

    def clean_text(text):
        """
        cleans text fast.
        """
        text = text.replace('\n', ' ').lower()#
        text = remove_ip_addr.sub('', text)
        text = replace_spaces.sub(' ',text)
        text = remove_special_chars.sub('',text)
        text = ' '.join([w for w in text.split() if not w in STOPWORDS])
        return text

    X_train = dataframe[column].fillna("missing")
    X_train = X_train.map(clean_text)

    # Dictionary of all words from train corpus with their counts.

    words_counts = {}
    for words in X_train:
        for word in words.split():
            if word not in words_counts:
                words_counts[word] = 1
            words_counts[word] += 1

    vocab_size = 50000
    top_words = sorted(words_counts, key=words_counts.get, reverse=True)[:vocab_size]

    text_join = ' '.join(top_words)

    picture_mask = plt.imread('test.png')

    wordcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',
                          width=1800,
                          height=1400,
                          mask=picture_mask
                ).generate(text_join)

    fig = plt.figure(figsize = (15, 15), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    ### This is where you save the fig or show the fig ######
    image_count = 0
    if verbose == 2:
        imgdata_list.append(save_image_data(fig, chart_format,
                            plot_name, depVar, mk_dir))
        image_count += 1
    if verbose <= 1:
        plt.show();
    ####### End of Pivot Plotting #############################
    return imgdata_list
################################################################################