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
# ipython inline magic shouldn't be needed because all plots are
# being displayed with plt.show() calls
get_ipython().magic('matplotlib inline')
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
#####################################################
class AutoViz_Class():
    """
        ##############################################################################
        #############       This is not an Officially Supported Google Product! ######
        ##############################################################################
        #Copyright 2019 Google LLC                                              ######
        #                                                                       ######
        #Licensed under the Apache License, Version 2.0 (the "License");        ######
        #you may not use this file except in compliance with the License.       ######
        #You may obtain a copy of the License at                                ######
        #                                                                       ######
        #    https://www.apache.org/licenses/LICENSE-2.0                        ######
        #                                                                       ######
        #Unless required by applicable law or agreed to in writing, software    ######
        #distributed under the License is distributed on an "AS IS" BASIS,      ######
        #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.#####
        #See the License for the specific language governing permissions and    ######
        #limitations under the License.                                         ######
        ##############################################################################
        ###########             AutoViz Class                                   ######
        ###########             by Ram Seshadri                                 ######
        ###########      AUTOMATICALLY VISUALIZE ANY DATA SET                   ######
        ###########            V3.0 6/15/19 Version                             ######
        ##############################################################################
        ##### AUTOVIZ PERFORMS AUTOMATIC VISUALIZATION OF ANY DATA SET WITH ONE CLICK.
        #####    Give it any input file (CSV, txt or json) and AV will visualize it.##
        ##### INPUTS:                                                            #####
        #####    A FILE NAME OR A DATA FRAME AS INPUT.                           #####
        ##### AutoViz will visualize any sized file using a statistically valid sample.
        #####  - COMMA is assumed as default separator in file. But u can change it.##
        #####  - Assumes first row as header in file but you can change it.      #####
        #####  - First instantiate an AutoViz class to  hold output of charts, plots.#
        #####  - Then call the Autoviz program with inputs as defined below.       ###
        ##############################################################################
    """
    def __init__(self):
        self.overall = {
        'name': 'overall',
        'plots': [],
        'heading': [],
        'subheading':[],  #"\n".join(subheading)
        'desc': [],  #"\n".join(subheading)
        'table1_title': "",
        'table1': [],
        'table2_title': "",
        'table2': []
            }  ### This is for overall description and comments about the data set
        self.scatter_plot = {
        'name': 'scatter',
        'heading': 'Scatter Plot of each Continuous Variable against Target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ##### This is for description and images for scatter plots ###
        self.pair_scatter = {
        'name': 'pair-scatter',
        'heading': 'Pairwise Scatter Plot of each Continuous Variable against other Continuous Variables',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': []  #"\n".join(desc)
        }   ##### This is for description and images for pairs of scatter plots ###
        self.dist_plot = {
        'name': 'distribution',
        'heading': 'Distribution Plot of Target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': []  #"\n".join(desc)
        } ##### This is for description and images for distribution plots ###
        self.pivot_plot = {
        'name': 'pivot',
        'heading': 'Pivot Plots of all Continuous Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        } ##### This is for description and images for pivot  plots ###
        self.violin_plot = {
        'name': 'violin',
        'heading': 'Violin Plots of all Continuous Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ##### This is for description and images for violin plots ###
        self.heat_map = {
        'name': 'heatmap',
        'heading': 'Heatmap of all Continuous Variables for target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }   ##### This is for description and images for heatmaps ###
        self.bar_plot = {
        'name': 'bar',
        'heading': 'Bar Plots of Average of each Continuous Variable by Target Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ##### This is for description and images for bar plots ###
        self.date_plot = {
        'name': 'time-series',
        'heading': 'Time Series Plots of Two Continuous Variables against a Date/Time Variable',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ######## This is for description and images for date time plots ###

    def add_plots(self,plotname,X):
        """
        This is a simple program to append the input chart to the right variable named plotname
        which is an attribute of class AV. So make sure that the plotname var matches an exact
        variable name defined in class AV. Otherwise, this will give an error.
        """
        if X is None:
            ### If there is nothing to add, leave it as it is.
            print("Nothing to add Plot not being added")
            pass
        else:
            eval('self.'+plotname+'["plots"].append(X)')

    def add_subheading(self,plotname,X):
        """
        This is a simple program to append the input chart to the right variable named plotname
        which is an attribute of class AV. So make sure that the plotname var matches an exact
        variable name defined in class AV. Otherwise, this will give an error.
        """
        if X is None:
            ### If there is nothing to add, leave it as it is.
            pass
        else:
            eval('self.'+plotname+'["subheading"].append(X)')

    def AutoViz(self, filename, sep=',', depVar='', dfte=None, header=0, verbose=0,
                            lowess=False,chart_format='svg',max_rows_analyzed=150000,
                                max_cols_analyzed=30):
        """
        ##############################################################################
        #############       This is not an Officially Supported Google Product! ######
        ##############################################################################
        #Copyright 2019 Google LLC                                              ######
        #                                                                       ######
        #Licensed under the Apache License, Version 2.0 (the "License");        ######
        #you may not use this file except in compliance with the License.       ######
        #You may obtain a copy of the License at                                ######
        #                                                                       ######
        #    https://www.apache.org/licenses/LICENSE-2.0                        ######
        #                                                                       ######
        #Unless required by applicable law or agreed to in writing, software    ######
        #distributed under the License is distributed on an "AS IS" BASIS,      ######
        #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.#####
        #See the License for the specific language governing permissions and    ######
        #limitations under the License.                                         ######
        ##############################################################################
        ###########             AutoViz Class                                   ######
        ###########             by Ram Seshadri                                 ######
        ###########      AUTOMATICALLY VISUALIZE ANY DATA SET                   ######
        ###########            V3.0 6/15/19 Version                             ######
        ##############################################################################
        ##### AUTOVIZ PERFORMS AUTOMATIC VISUALIZATION OF ANY DATA SET WITH ONE CLICK.
        #####    Give it any input file (CSV, txt or json) and AV will visualize it.##
        ##### INPUTS:                                                            #####
        #####    A FILE NAME OR A DATA FRAME AS INPUT.                           #####
        ##### AutoViz will visualize any sized file using a statistically valid sample.
        #####  - max_rows_analyzed = 150000 ### this limits the max number of rows ###
        #####           that is used to display charts                             ###
        #####  - max_cols_analyzed = 30  ### This limits the number of continuous  ###
        #####           vars that can be analyzed                                 ####
        #####  - COMMA is assumed as default separator in file. But u can change it.##
        #####  - Assumes first row as header in file but you can change it.      #####
        #####  - First instantiate an AutoViz class to  hold output of charts, plots.#
        #####  - Then call the Autoviz program with inputs as defined below.       ###
        ##############################################################################
        ##### This is the main calling program in AV. It will call all the load, #####
        ####  display and save rograms that are currently outside AV. This program ###
        ####  will draw scatter and other plots for the input data set and then   ####
        ####  call the correct variable name with add_plots function and send in  ####
        ####  the chart created by that plotting program, for example, scatter   #####
        ####  You have to make sure that add_plots function has the exact name of ####
        ####  the variable defined in the Class AV. If not, this will give an error.##
        ####  If verbose=0: it does not print any messages and goes into silent mode##
        ####  This is the default.                                               #####
        ####  If verbose=1, it will print messages on the terminal and also display###
        ####  charts on terminal                                                 #####
        ####  If verbose=2, it will print messages but will not display charts,  #####
        ####  it will simply save them.                                          #####
        ##############################################################################
        """
        max_num_cols_analyzed = min(25,int(max_cols_analyzed*0.6))
        start_time = time.time()
        try:
            dft, depVar,IDcols,bool_vars,cats,continuous_vars,discrete_string_vars,date_vars,classes,problem_type = classify_print_vars(
                                                filename,sep,max_rows_analyzed, max_cols_analyzed,
                                                depVar,dfte,header,verbose)
        except:
            print('Not able to read or load file. Please check your inputs and try again...')
            return None
        if depVar == None or depVar == '':
         ##### This is when No dependent Variable is given #######
            try:
                svg_data = draw_pair_scatters(dft,continuous_vars,problem_type,verbose,chart_format,
                                                depVar,classes,lowess)
                self.add_plots('pair_scatter',svg_data)
            except Exception as e:
                print(e)
                print('Could not draw Pair Scatter Plots')
            try:
                svg_data = draw_distplot(dft, bool_vars+cats+continuous_vars,verbose,chart_format,problem_type)
                self.add_plots('dist_plot',svg_data)
            except:
                print('Could not draw Distribution Plot')
            try:
                svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type)
                self.add_plots('violin_plot',svg_data)
            except:
                print('Could not draw Violin Plot')
            try:
                svg_data = draw_heatmap(dft, continuous_vars, verbose,chart_format, date_vars, depVar)
                self.add_plots('heat_map',svg_data)
            except:
                print('Could not draw Heat Map')
            if date_vars != [] and len(continuous_vars)<=max_num_cols_analyzed:
                try:
                    svg_data = draw_date_vars(dft,depVar,date_vars,
                                              continuous_vars,verbose,chart_format,problem_type)
                    self.add_plots('date_plot',svg_data)
                except:
                    print('Could not draw Date Vars')
            if len(cats) <= 10 and len(continuous_vars)<=max_num_cols_analyzed:
                try:
                    svg_data = draw_barplots(dft,cats+bool_vars,continuous_vars, problem_type,
                                    verbose,chart_format,depVar)
                    self.add_plots('bar_plot',svg_data)
                except:
                    print('Could not draw Bar Plots')
            else :
                 print('Number of Categorical and Continuous Vars exceeds limit, hence no Bar Plots')
            print('Time to run AutoViz (in seconds) = %0.3f' %(time.time()-start_time))
            if verbose == 1:
                print('\n ###################### VISUALIZATION Completed ########################')
        else:
            if problem_type=='Regression':
                ############## This is a Regression Problem #################
                try:
                    svg_data = draw_scatters(dft,
                                    continuous_vars,verbose,chart_format,problem_type,depVar,classes,lowess)
                    self.add_plots('scatter_plot',svg_data)
                except Exception as e:
                    print("Exception Drawing Scatter Plots")
                    print(e)
                    traceback.print_exc()
                    print('Could not draw Scatter Plots')
                try:
                    svg_data = draw_pair_scatters(dft,continuous_vars,problem_type,verbose,chart_format,
                                                    depVar,classes,lowess)
                    self.add_plots('pair_scatter',svg_data)
                except:
                    print('Could not draw Pair Scatter Plots')
                try:
                    if type(depVar) == str:
                        othernums = [x for x in continuous_vars if x not in [depVar]]
                    else:
                        othernums = [x for x in continuous_vars if x not in depVar]
                    if len(othernums) >= 1:
                        svg_data = draw_distplot(dft, bool_vars+cats+continuous_vars,verbose,chart_format,problem_type,depVar,classes)
                        self.add_plots('dist_plot',svg_data)
                    else:
                        print('No continuous var in data set: hence no distribution plots')
                except:
                    print('Could not draw Distribution Plots')
                try:
                    svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type)
                    self.add_plots('violin_plot',svg_data)
                except:
                    print('Could not draw Violin Plots')
                try:
                    svg_data = draw_heatmap(dft,
                                        continuous_vars, verbose,chart_format, date_vars, depVar,problem_type)
                    self.add_plots('heat_map',svg_data)
                except:
                    print('Could not draw Heat Maps')
                if date_vars != [] and len(continuous_vars)<=max_num_cols_analyzed:
                    try:
                        svg_data = draw_date_vars(
                            dft,depVar,date_vars,continuous_vars,verbose,chart_format,problem_type)
                        self.add_plots('date_plot',svg_data)
                    except:
                        print('Could not draw Time Series plots')
                if len(cats) <= 10 and len(continuous_vars) <= max_num_cols_analyzed:
                    try:
                        svg_data = draw_pivot_tables(dft,cats+bool_vars,
                                    continuous_vars,problem_type,verbose,chart_format,depVar)
                        self.add_plots('pivot_plot',svg_data)
                    except:
                        print('Could not draw Pivot Charts against Dependent Variable')
                    try:
                        svg_data = draw_barplots(dft,cats+bool_vars,continuous_vars,problem_type,verbose,
                                                    chart_format,depVar)
                        self.add_plots('bar_plot',svg_data)
                        #self.add_plots('bar_plot',None)
                        print('All Plots done')
                    except:
                        print('Could not draw Bar Charts')
                else:
                    print ('Number of Cat and Continuous Vars exceeds %d, hence no Pivot Tables' %max_cols_analyzed)
                print('Time to run AutoViz (in seconds) = %0.3f' %(time.time()-start_time))
                if verbose == 1:
                    print('\n ###################### VISUALIZATION Completed ########################')
            else :
                ############ This is a Classification Problem ##################
                try:
                    svg_data = draw_scatters(dft,continuous_vars,
                                             verbose,chart_format,problem_type,depVar, classes,lowess)
                    self.add_plots('scatter_plot',svg_data)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print("Exception Drawing Scatter Plots")
                    print('Could not draw Scatter Plots')
                try:
                    svg_data = draw_pair_scatters(dft,continuous_vars,
                                                  problem_type,verbose,chart_format,depVar,classes,lowess)
                    self.add_plots('pair_scatter',svg_data)
                except:
                    print('Could not draw Pair Scatter Plots')
                try:
                    if type(depVar) == str:
                        othernums = [x for x in continuous_vars if x not in [depVar]]
                    else:
                        othernums = [x for x in continuous_vars if x not in depVar]
                    if len(othernums) >= 1:
                        svg_data = draw_distplot(dft, bool_vars+cats+continuous_vars,verbose,chart_format,
                                                problem_type,depVar,classes)
                        self.add_plots('dist_plot',svg_data)
                    else:
                        print('No continuous var in data set: hence no distribution plots')
                except:
                    print('Could not draw Distribution Plots')
                try:
                    svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type)
                    self.add_plots('violin_plot',svg_data)
                except:
                    print('Could not draw Violin Plots')
                try:
                    svg_data = draw_heatmap(dft, continuous_vars,
                                            verbose,chart_format, date_vars, depVar,problem_type,classes)
                    self.add_plots('heat_map',svg_data)
                except:
                    print('Could not draw Heat Maps')
                if date_vars != [] and len(continuous_vars)<=max_num_cols_analyzed:
                    try:
                        svg_data = draw_date_vars(dft,depVar,date_vars,
                                                  continuous_vars,verbose,chart_format,problem_type)
                        self.add_plots('date_plot',svg_data)
                    except:
                        print('Could not draw Time Series plots')
                if len(cats) <= 10 and len(continuous_vars)<=max_num_cols_analyzed:
                    try:
                        svg_data = draw_pivot_tables(
                            dft,cats+bool_vars,continuous_vars,problem_type,verbose,chart_format,depVar,classes)
                        self.add_plots('pivot_plot',svg_data)
                    except:
                        print('Could not draw Pivot Charts against Dependent Variable')
                    try:
                        if len(classes) > 2:
                            svg_data = draw_barplots(dft,cats+bool_vars,continuous_vars,problem_type,
                                            verbose,chart_format,depVar, classes)
                            self.add_plots('bar_plot',svg_data)
                        else:
                            self.add_plots('bar_plot',None)
                        print('All plots done')
                        pass
                    except:
                        if verbose == 1:
                            print('Could not draw Bar Charts')
                        pass
                else:
                    print('Number of Cat and Continuous Vars exceeds %d, hence no Pivot or Bar Charts' %max_cols_analyzed)
                print('Time to run AutoViz (in seconds) = %0.3f' %(time.time()-start_time))
                if verbose == 1:
                    print ('\n ###################### VISUALIZATION Completed ########################')
        return dft

######## This is where we store the image data in a dictionary with a list of images #########
def save_image_data(fig, image_count, chart_format):
    if chart_format == 'svg':
        ###### You have to add these lines to each function that creates charts currently ##
        imgdata = io.StringIO()
        fig.savefig(imgdata, format=chart_format)
        imgdata.seek(0)
        svg_data = imgdata.getvalue()
        return svg_data
    else:
        ### You have to do it slightly differently for PNG and JPEG formats
        imgdata = BytesIO()
        fig.savefig(imgdata, format=chart_format, bbox_inches='tight', pad_inches=0.0)
        imgdata.seek(0)
        figdata_png = base64.b64encode(imgdata.getvalue())
        return figdata_png

#### This module analyzes a dependent Variable and finds out whether it is a
#### Regression or Classification type problem
def analyze_problem_type(train, targ,verbose=0) :
    if train[targ].dtype != 'int64' and train[targ].dtype != float :
        if len(train[targ].unique()) == 2:
            if verbose == 1:
                print('"\n ################### Binary-Class VISUALIZATION Started ##################### " ')
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 1 and len(train[targ].unique()) <= 15:
                model_class = 'Multi_Classification'
                if verbose == 1:
                    print('"\n ################### Multi-Class VISUALIZATION Started ######################''')
    elif train[targ].dtype == 'int64' or train[targ].dtype == float :
        if len(train[targ].unique()) == 2:
            if verbose == 1:
                print('"\n ################### Binary-Class VISUALIZATION Started ##################### " ')
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 1 and len(train[targ].unique()) <= 15:
                model_class = 'Multi_Classification'
                if verbose == 1:
                    print('"\n ################### Multi-Class VISUALIZATION Started ######################''')
        else:
            model_class = 'Regression'
            if verbose == 1:
                print('"\n ################### Regression VISUALIZATION Started ######################''')
    elif train[targ].dtype == object:
            if len(train[targ].unique()) > 1 and len(train[targ].unique()) <= 2:
                model_class = 'Binary_Classification'
                if verbose == 1:
                    print('"\n ################### Binary-Class VISUALIZATION Started ##################### " ')
            else:
                model_class = 'Multi_Classification'
                if verbose == 1:
                    print('"\n ################### Multi-Class VISUALIZATION Started ######################''')
    elif train[targ].dtype == bool:
                model_class = 'Binary_Classification'
                if verbose == 1:
                    print('"\n ################### Binary-Class VISUALIZATION Started ######################''')
    elif train[targ].dtype == 'int64':
        if len(train[targ].unique()) == 2:
            if verbose == 1:
                print('"\n ################### Binary-Class VISUALIZATION Started ##################### " ')
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 1 and len(train[targ].unique()) <= 25:
                model_class = 'Multi_Classification'
                if verbose == 1:
                    print('"\n ################### Multi-Class VISUALIZATION Started ######################''')
        else:
            model_class = 'Regression'
            if verbose == 1:
                print('"\n ################### Regression VISUALIZATION Started ######################''')
    else :
        if verbose == 1:
            print('\n ###################### REGRESSION VISUALIZATION Started #####################')
        model_class = 'Regression'
    return model_class

# Pivot Tables are generally meant for Categorical Variables on the axes
# and a Numeric Column (typically the Dep Var) as the "Value" aggregated by Sum.
# Let's do some pivot tables to capture some meaningful insights
def draw_pivot_tables(dft,cats,nums,problem_type,verbose,chart_format,depVar='', classes=None):
    cats = list(set(cats))
    dft = dft[:]
    cols = 2
    cmap = plt.get_cmap('jet')
    #### For some reason, the cmap colors are not working #########################
    colors = cmap(np.linspace(0, 1, len(cats)))
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgkbyr')
    colormaps = ['summer', 'rainbow','viridis','inferno','magma','jet','plasma']
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
    height_size = 4
    if problem_type == 'Regression' or depVar==None or depVar=='' or depVar==[]:
        image_count = 0
        ###### This is meant for Regression problems where the DepVar is of Continuous Type
        if noplots%cols == 0:
            rows = noplots/cols
        else:
            rows = (noplots/cols)+1
        fig = plt.figure(figsize=(width_size,rows*height_size))
        fig.suptitle('Bar Plots of each Continuous Var by %s' %depVar, fontsize=20,y=1.08)
        k = 1
        for i,color in zip(range(len(cats)), colors) :
            if len(dft[cats[i]].unique() )>= categorylimit:
                plt.subplot(rows,cols,k)
                ax1 = plt.gca()
                dft.groupby(cats[i])[depVar].mean().sort_values(ascending=False)[:displaylimit].plot(kind='bar',
                    title='Average %s vs. %s (Descending) ' %(depVar, cats[i]),ax=ax1,
                    colormap=random.choice(colormaps))
                for p in ax1.patches:
                    ax1.annotate(str(round(p.get_height(),2)),(round(p.get_x()*1.01,2),round(p.get_height()*1.01,2)))
                k += 1
            else:
                plt.subplot(rows,cols,k)
                ax1 = plt.gca()
                dft.groupby(cats[i])[depVar].mean().sort_values(ascending=False).plot(kind='bar',
                    title='Average %s vs. %s (Descending) ' % (depVar, cats [i]), ax=ax1,
                    colormap=random.choice(colormaps))
                for p in ax1.patches:
                    ax1.annotate(str(round(p.get_height(),2)),(round(p.get_x()*1.01,2), round(p.get_height()*1.01,2)))
                k += 1
        fig.tight_layout();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
    else:
        ###### This is meant for Classification problems where the DepVar is an Object type
        image_count = 0
        chunksize = 20
        N = len(nums)
        lst=[]
        noplots=int((N**2-N)/2)
        dicti = {}
        if len(nums) == 1:
            pass
        ls_cats = []
        for each_cat in cats:
            if len(dft[dft[depVar]==classes[0]][each_cat].value_counts()) == 1:
                pass
            else:
                ls_cats.append(each_cat)
        if len(ls_cats) <= 2:
            cols = 2
            noplots = len(nums)
            rows = int((noplots/cols)+0.99)
            counter = 1
            fig = plt.figure(figsize=(width_size,rows*height_size))
            fig.suptitle('Plots of each Continuous Var by %s' %depVar, fontsize=20,y=1.08)
            plt.subplots_adjust(hspace=0.5)
            for eachpred,color3 in zip(nums,colors):
                ### Be very careful with the next line. It should be singular "subplot" ##
                ##### Otherwise, if you use the plural version "subplots" it has a different meaning!
                plt.subplot(rows,cols,counter)
                ax1 = plt.gca()
                dft[[eachpred,depVar]].groupby(depVar).mean().plot(kind='bar', ax=ax1, colors=color3)
                for p in ax1.patches:
                    ax1.annotate(str(round(p.get_height(),2)), (round(p.get_x()*1.01,2), round(p.get_height()*1.01,2)))
                ax1.set_title('Average of %s by %s' %(eachpred,depVar))
                plt.legend(loc='best')
                counter += 1
            fig.tight_layout();
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, image_count, chart_format))
                image_count += 1
        else:
            N = len(ls_cats)
            combos = combinations(ls_cats,2)
            noplots = int((N**2-N)/2)
            rows = int((noplots/cols)+0.99)
            num_plots = len(classes)*noplots/2.0
            if verbose == 1:
                print('No. of Bar Plots = %s' %num_plots)
            rows = int((num_plots/cols)+0.99)
            fig = plt.figure(figsize=(width_size,rows*height_size))
            target_vars = dft[depVar].unique()
            if len(classes) == 2:
                func = 'np.mean'
                func_keyword = 'Average'
            else:
                func = 'len'
                func_keyword = 'Number'
            fig.suptitle('Plots of %s of each Continuous Var by %s' %(func_keyword,depVar),fontsize=20,y=1.08)
            plotcounter = 1
            if dft[depVar].dtype == object:
                dft[depVar] = dft[depVar].factorize()[0]
            for (var1, var2) in combos:
                if len(classes) == 2:
                    plt.subplot(rows, cols, plotcounter)
                    ax1 = plt.gca()
                    try:
                        #pd.pivot_table(data=dft,columns=var1, index=var2, values=depVar, aggfunc=eval(func))
                        if func == 'np.mean':
                            dft[[var1,var2,depVar]].groupby([var1,var2])[depVar].mean().sort_values()[
                                        :chunksize].plot(
                                    kind='bar', colormap=random.choice(colormaps),ax=ax1)
                        else:
                            dft[[var1,var2,depVar]].groupby([var1,var2])[depVar].size().sort_values()[
                                        :chunksize].plot(
                                    kind='bar', colormap=random.choice(colormaps),ax=ax1)
                        ax1.set_title('Percentage of %s grouped by %s and %s' %(depVar,var1,var2))
                    except:
                        dft.pivot(columns=var1, values=var2).plot(kind='bar', colormap='plasma',ax=ax1)
                        plt.xlabel(var2)
                        plt.ylabel(depVar)
                        ax1.set_title('Percentage of %s grouped by %s and %s' %(depVar,var1,var2))
                    plt.legend()
                    plotcounter += 1
                else:
                    #### Fix color in all Scatter plots using this trick:
                    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgkbyr')
                    for target_var, color_val,class_label in zip(target_vars, colors,classes):
                        plt.subplot(rows, cols, plotcounter)
                        ax1 = plt.gca()
                        dft_target = dft[dft[depVar]==target_var]
                        try:
                            pd.pivot_table(data=dft_target,columns=var1, index=var2, values=depVar,
                                           aggfunc=eval(func)).plot(kind='bar', colormap='plasma',ax=ax1)
                        except:
                            dft.pivot(columns=var1, values=var2).plot(kind='bar', colormap='plasma',ax=ax1)
                        plt.xlabel(var2)
                        plt.ylabel(target_var)
                        ax1.set_title('%s of %s grouped by %s and %s' %(func_keyword,class_label,var2,var1))
                        plt.legend()
                        plotcounter += 1
            fig.tight_layout();
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, image_count, chart_format))
                image_count += 1
            if verbose == 1:
                plt.show();
    ####### End of Pivot Plotting #############################
    return imgdata_list

# In[ ]:
# SCATTER PLOTS ARE USEFUL FOR COMPARING NUMERIC VARIABLES
def draw_scatters(dfin,nums,verbose,chart_format,problem_type,dep=None, classes=None, lowess=False):
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
        if verbose == 1:
            print('Using Lowess Smoothing. This might take a few minutes for large data sets...')
        lowess = True
        x_est = None
        transparent = 0.6
        bubble_size = 100
    if verbose == 1:
        x_est = np.mean
    N = len(nums)
    cols = 2
    width_size = 15
    height_size = 4
    if dep == None or dep == '':
        image_count = 0
        ##### This is when no Dependent Variable is given ###
        ### You have to do a Pair-wise Scatter Plot of all Continuous Variables ####
        combos = combinations(nums, 2)
        noplots = int((N**2-N)/2)
        print('Number of Scatter Plots = %d' %(noplots+N))
        rows = int((noplots/cols)+0.99)
        fig = plt.figure(figsize=(width_size,rows*height_size))
        for (var1,var2), plotcounter, color_val in zip(combos, range(1,noplots+1), colors):
            ### Be very careful with the next line. It should be singular "subplot" ##
            ##### Otherwise, if you use the plural version "subplots" it has a different meaning!
            plt.subplot(rows,cols,plotcounter)
            if lowess:
                sns.regplot(x=dft[var1], y = dft[var2], lowess=lowess, color=color_val, ax=plt.gca())
            else:
                sns.scatterplot(x=dft[var1], y=dft[var2], ax=plt.gca(), paletter='dark',color=color_val)
            plt.xlabel(var1)
            plt.ylabel(var2)
        fig.suptitle('Pair-wise Scatter Plot of all Continuous Variables',fontsize=20,y=1.08)
        fig.tight_layout();
        if verbose == 1:
            plt.show();
        #### Keep it at the figure level###
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
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
                sns.scatterplot(x=dft[dep], y=dft[num], ax=plt.gca(), palette='dark',color=color_val)
            plt.xlabel(num)
            plt.ylabel(dep)
        fig.suptitle('Scatter Plot of each Continuous Variable against Target Variable', fontsize=20,y=1.08)
        fig.tight_layout();
        if verbose == 1:
            plt.show();
        #### Keep it at the figure level###
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
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
        plt.suptitle('Scatter Plot of Continuous Variable vs Target (jitter=%0.2f)' %jitter, fontsize=20,y=1.08)
        fig.tight_layout();
        if verbose == 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
    ####### End of Scatter Plots ######
    return imgdata_list

# PAIR SCATTER PLOTS ARE NEEDED ONLY FOR CLASSIFICATION PROBLEMS IN NUMERIC VARIABLES
def draw_pair_scatters(dfin,nums,problem_type, verbose,chart_format, dep=None, classes=None, lowess=False):
    """
    ### This is where you plot a pair-wise scatter plot of Independent Variables against each other####
    """
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
    if verbose == 1:
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
        fig.suptitle('Pair-wise Scatter Plot of all Continuous Variables', fontsize=20,y=1.08)
        fig.tight_layout();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
        if verbose == 1:
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
        fig.suptitle('Scatter Plot of each Continuous Variable against Target Variable', fontsize=20,y=1.08)
        fig.tight_layout();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
        if verbose == 1:
            plt.show();
    ####### End of Pair Scatter Plots ######
    return imgdata_list

#Bar Plots are for 2 Categoricals and One Numeric (usually Dep Var)
def draw_barplots(dft,cats,conti,problem_type,verbose,chart_format,dep='', classes=None):
    #### Category limit within a variable ###
    cats = cats[:]
    cat_limit = 10
    width_size = 15
    height_size = 4
    conti = list_difference(conti,dep)
    #### Remove Floating Point Categorical Vars from this list since they Error when Bar Plots are drawn
    cats = [x for x in cats if dft[x].dtype != float]
    dft = dft[:]
    N = len(cats)
    if N==0:
        print('No categorical or boolean vars in data set. Hence no bar charts.')
        return None
    cmap = plt.get_cmap('jet')
    ### Not sure why the cmap doesn't work and gives an error in some cases #################
    colors = cmap(np.linspace(0, 1, len(conti)))
    colors = cycle('gkbyrcmgkbyrcmgkbyrcmgkbyr')
    colormaps = ['plasma','viridis','inferno','magma']
    imgdata_list = list()
    nums = [x for x in list (dft) if dft[x].dtype=='float64' and x not in [dep]+cats]
    for k in range(len(cats)):
        image_count = 0
        N = len(conti)
        order= dft[cats[k]].unique().tolist()
        nocats = len(order)
        if nocats >= 100:
            chunksize=100
            cols = 1
        else:
            if nocats >= 25:
                chunksize = 20
                cols = 2
            else:
                chunksize = cat_limit
                cols = 2
        if len(cats) == 0:
            noplots = len(conti)*cols
        else:
            noplots=len(conti)*len(cats)*cols
        if cols==2:
            if noplots%cols == 0:
                rows = noplots/cols
            else:
                rows = (noplots/cols)+1
        else:
            rows = copy.deepcopy(noplots)
        if rows >= 50:
            rows = 50
        stringlimit = 25
        if dep==None or dep == '':
            ########## This is when no Dependent Variable is Given ######
            fig = plt.figure(figsize=(width_size,rows*height_size))
            kadd = 1
            for each_conti,color in zip(conti,colors):
                plt.subplot(rows,cols,kadd)
                ax1 = plt.gca()
                dft.groupby(cats[k])[each_conti].mean().sort_values(ascending=False)[:chunksize].plot(
                    kind='bar',ax=ax1, color=color)
                ax1.set_title('Average %s by %s (Descending)' %(each_conti, cats[k]))
                if dft[cats[k]].dtype == object:
                    labels = dft.groupby(cats[k])[each_conti].mean().sort_values(
                        ascending=False)[:chunksize].index.str[:stringlimit].tolist()
                    ax1.set_xticklabels(labels)
                kadd += 1
                if verbose == 2:
                    imgdata_list.append(save_image_data(fig, image_count, chart_format))
                    image_count += 1
                ### This code is redundant unless number of levels in category are >20
                if dft[cats[k]].nunique() > chunksize:
                    plt.subplot(rows,cols,kadd)
                    ax1 = plt.gca()
                    dft.groupby(cats[k])[each_conti].mean().sort_values(
                        ascending=True)[:chunksize].plot(kind='bar',ax=ax1, color=color)
                    if dft[cats[k]].dtype == object:
                        labels = dft.groupby(cats [k])[each_conti].mean().sort_values(
                            ascending=True)[:chunksize].index.str[:stringlimit].tolist()
                        ax1.set_xticklabels(labels)
                    ax1.set_title('Average %s by %s (Ascending)' %(each_conti,cats[k]))
                    kadd += 1
            fig.tight_layout();
            if verbose == 1:
                plt.show();
            ###########
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, image_count, chart_format))
                image_count += 1
        elif problem_type == 'Regression':
            ########## This is for Regression Problems only ######
            fig = plt.figure(figsize=(width_size,rows*height_size))
            N = len(conti)
            noplots=int((N**2-N)/4)
            kadd = 1
            for each_conti,color in zip(conti,colors):
                if len(dft[cats[k]].value_counts()) < 20:
                    plt.subplot(rows,cols,kadd)
                    ax1 = plt.gca()
                    dft.groupby(cats[k])[each_conti].mean().sort_values(
                        ascending=False)[:chunksize].plot(kind='bar',ax=ax1,
                        colormap=random.choice(colormaps))
                    for p in ax1.patches:
                        ax1.annotate(str(round(p.get_height(),2)),(round(p.get_x()*1.01,2), round(p.get_height()*1.01,2)))
                    if dft[cats[k]].dtype == object:
                        labels = dft.groupby(cats[k])[each_conti].mean().sort_values(
                            ascending= True)[:chunksize].index.str[:stringlimit].tolist()
                        ax1.set_xticklabels(labels)
                    ax1.set_title('Average %s by %s (Descending)' %(each_conti,cats[k]))
                    kadd += 1
                else:
                    ### This code is redundant unless number of levels in category are >20
                    plt.subplot(rows,cols,kadd)
                    ax1 = plt.gca()
                    dft.groupby(cats[k])[each_conti].mean().sort_values(
                        ascending=True)[:chunksize].plot(kind='bar',ax=ax1,
                        colormap=random.choice(colormaps))
                    if dft[cats[k]].dtype == object:
                        labels = dft.groupby(cats[k])[each_conti].mean().sort_values(
                            ascending= True)[:chunksize].index.str[:stringlimit].tolist()
                        ax1.set_xticklabels(labels)
                    ax1.set_title('Mean %s by %s (Ascending)' %(each_conti,cats[k]))
                    kadd += 1
            fig.tight_layout();
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, image_count, chart_format))
                image_count += 1
            elif verbose == 1:
                plt.show();
        else:
            ########## This is for Classification Problems ######
            image_count = 0
            target_vars = dft[dep].unique()
            noplots = len(conti)*cols
            kadd = 1
            fig = plt.figure(figsize=(width_size,rows*height_size))
            if len(nums) == 0:
                for each_conti,color3 in zip(conti,colors):
                    plt.subplot(rows,cols,kadd)
                    ax1 = plt.gca()
                    dft.groupby(cats[k])[each_conti].mean().sort_values(
                        ascending=False).plot(kind='bar',ax=ax1,
                                                       color=color3)
                    ax1.set_title('Average %s by %s (Descending)' %(each_conti,cats[k]))
                    kadd += 1
            else:
                conti = copy.deepcopy(nums)
                for each_conti in conti:
                    plt.subplot(rows,cols,kadd)
                    ax1 = plt.gca()
                    dft.groupby([dep, cats[k]])[each_conti].mean().sort_values(
                        ascending=False).unstack().plot(kind='bar',ax=ax1,
                                                       colormap=random.choice(colormaps))
                    ax1.set_title('Average %s by %s (Descending)' %(each_conti,cats[k]))
                    kadd += 1
            fig.tight_layout();
            fig.suptitle('Bar Plots of Continuous Variables by %s' %cats[k], fontsize=20, y=1.08)
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, image_count, chart_format))
                image_count += 1
            elif verbose == 1:
                plt.show();
    return imgdata_list
    ############## End of Bar Plotting ##########################################

##### Draw a Heatmap using Pearson Correlation #########################################
def draw_heatmap(dft, conti, verbose,chart_format,datevars=[], dep=None,
                                    modeltype='Regression',classes=None):
    ### Test if this is a time series data set, then differene the continuous vars to find
    ###  if they have true correlation to Dependent Var. Otherwise, leave them as is
    width_size = 3
    height_size = 2
    if len(conti) <= 1:
        return
    if isinstance(dft.index, pd.DatetimeIndex) :
        dft = dft[:]
        timeseries_flag = True
        pass
    else:
        dft = dft[:]
        try:
            dft.index = pd.to_datetime(dft.pop(datevars[0]),infer_datetime_format=True)
            timeseries_flag = True
        except:
            if verbose == 1 and len(datevars) > 0:
                print('No date vars could be found or %s could not be indexed.' %datevars)
            elif verbose == 1 and len(datevars) == 0:
                print('No date vars could be found in data set')
            timeseries_flag = False
    # Add a column: the color depends on target variable but you can use whatever function
    imgdata_list = list()
    if modeltype != 'Regression':
        ########## This is for Classification problems only ###########
        if dft[dep].dtype == object or dft[dep].dtype == np.int64:
            dft[dep] = dft[dep].factorize()[0]
        image_count = 0
        N = len(conti)
        target_vars = dft[dep].unique()
        fig = plt.figure(figsize=(min(N*width_size,20),min(N*height_size,20)))
        plotc = 1
        #rows = len(target_vars)
        rows = 1
        cols = 1
        if timeseries_flag:
            dft_target = dft[[dep]+conti].diff()
        else:
            dft_target = dft[:]
        dft_target[dep] = dft[dep]
        corr = dft_target.corr()
        plt.subplot(rows, cols, plotc)
        ax1 = plt.gca()
        sns.heatmap(corr, annot=True,ax=ax1)
        plotc += 1
        if timeseries_flag:
            plt.title('Time Series: Heatmap of all Differenced Continuous vars for target = %s' %dep)
        else:
            plt.title('Heatmap of all Continuous Variables for target = %s' %dep)
        fig.tight_layout();
        if verbose == 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
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
            plt.title('Time Series Data: Heatmap of Differenced Continuous vars including target = %s' %dep)
        else:
            plt.title('Heatmap of all Continuous Variables including target = %s' %dep)
        fig.tight_layout();
        if verbose == 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
    return imgdata_list
    ############# End of Heat Maps ##############

##### Draw the Distribution of each variable using Distplot
##### Must do this only for Continuous Variables
def draw_distplot(dft, conti,verbose,chart_format,problem_type,dep=None, classes=None):
    #### Since we are making changes to dft and classes, we will be making copies of it here
    dft = dft[:]
    classes = copy.deepcopy(classes)
    colors = cycle('brycgkbyrcmgkbyrcmgkbyrcmgkbyr')
    cols = 2
    imgdata_list = list()
    width_size = 15
    height_size = 5
    gap = 0.3
    if dep==None or dep=='' or problem_type == 'Regression':
        image_count = 0
        transparent = 0.7
        ######### This is for cases where there is No Target or Dependent Variable ########
        if problem_type == 'Regression':
            if isinstance(dep,list):
                conti += dep
            else:
                conti += [dep]
        noplots = len(conti)
        rows = int((noplots/cols)+0.99 )
        ### Be very careful with the next line. we have used the plural "subplots" ##
        ## In this case, you have ax as an array and you have to use (row,col) to get each ax!
        fig = plt.figure(figsize=(width_size,rows*height_size))
        fig.subplots_adjust(hspace=gap) ### This controls the space betwen rows
        for k, color2 in zip(range(noplots),colors):
            #print('Iteration %s' %k)
            conti_iter = conti[k]
            if dft[conti_iter].dtype == float or dft[conti_iter].dtype==np.int32 or dft[conti_iter].dtype==np.int64:
                if dft[conti_iter].nunique() <= 25:
                    chart_type = 'bar'
                else:
                    chart_type = 'float'
            elif dft[conti_iter].dtype == object and dft[conti_iter].nunique() <= 25:
                chart_type = 'bar'
            else:
                chart_type = 'bar'
            if chart_type == 'float':
                if dft[conti_iter].min() == 0.0:
                    hist_bins = 25
                elif dft[conti_iter].max()/dft[conti_iter].min() > 50 or dft[conti_iter].max()-dft[conti_iter].min()  > 50:
                    hist_bins = 50
                else:
                    hist_bins = 30
                plt.subplot(rows, cols, k+1)
                ax1 = plt.gca()
                #ax2 = ax1.twiny()
                if len(dft[dft[conti_iter]<0]) > 0:
                    ### If there are simply neg numbers in the column, better to skip the Log...
                    #dft[conti_iter].hist(bins=hist_bins, ax=ax1, color=color2,label='%s' %conti_iter,
                    #               )
                    sns.distplot(dft[conti_iter],
                        hist=False, kde=True,label='%s' %conti_iter,
                        bins=hist_bins, ax= ax1,hist_kws={'alpha':transparent},
                        color=color2)
                    ax1.legend(loc='upper right')
                    ax1.set_xscale('linear')
                    ax1.set_xlabel('Linear Scale')
                    #ax1.set_title('%s Distribution (No Log transform since negative numbers)' %conti_iter,
                    #                            loc='center',y=1.18)
                elif len(dft[dft[conti_iter]==0]) > 0:
                    ### If there are only zeros numbers in the column, you can do log transform by adding 1...
                    #dft[conti_iter].hist(bins=hist_bins, ax=ax1, color=color2,label='before log transform'
                    #                    )
                    sns.distplot(dft[conti_iter],
                        hist=False, kde=True,label='%s' %conti_iter,hist_kws={'alpha':transparent},
                        bins=hist_bins, ax= ax1,
                        color=color2)
                    #np.log(dft[conti_iter]+1).hist(bins=hist_bins, ax=ax2, color=next(colors),
                    #    alpha=transparent, label='after log transform',bw_method=3)
                    #sns.distplot(np.log10(dft[conti_iter]+1),
                    #    hist=False, kde=True,hist_kws={'alpha':transparent},
                    #    bins=hist_bins, ax= ax2,label='after potential log transform',
                    #    color=next(colors))
                    ax1.legend(loc='upper right')
                    #ax2.legend(loc='upper left')
                    ax1.set_xscale('linear')
                    #ax2.set_xscale('log')
                    ax1.set_xlabel('Linear Scale')
                    #ax2.set_xlabel('Log Scale')
                    #ax1.set_title('%s Distribution and potential Log Transform' %conti_iter, loc='center',y=1.18)
                else:
                    ### if there are no zeros and no negative numbers then it is a clean data ########
                    #dft[conti_iter].hist(bins=hist_bins, ax=ax1, color=color2,label='before log transform',
                    #                    bw_method=3)
                    sns.distplot(dft[conti_iter],
                        hist=False, kde=True,label='%s' %conti_iter,
                        bins=hist_bins, ax= ax1,hist_kws={'alpha':transparent},
                        color=color2)
                    #np.log(dft[conti_iter]).fillna(0).hist(bins=hist_bins, ax=ax2, color=next(colors),
                    #    alpha=transparent, label='after log transform',bw_method=3)
                    #sns.distplot(np.log10(dft[conti_iter]),
                    #    hist=False, kde=True,label='after potential log transform',
                    #    bins=hist_bins, ax= ax2,hist_kws={'alpha':transparent},
                    #    color=next(colors))
                    ax1.legend(loc='upper right')
                    #ax2.legend(loc='upper left')
                    ax1.set_xscale('linear')
                    #ax2.set_xscale('log')
                    ax1.set_xlabel('Linear Scale')
                    #ax2.set_xlabel('Log Scale')
                    #ax1.set_title('%s Distribution and potential Log Transform' %conti_iter, loc='center',y=1.18)
            else:
                plt.subplot(rows, cols, k+1)
                ax1 = plt.gca()
                dft[conti_iter].value_counts().plot(kind='bar',ax=ax1,label='%s' %conti_iter)
                ax1.set_title('%s Distribution' %conti_iter, loc='center',y=1.18)
        fig.tight_layout();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
        fig.suptitle('Histograms (KDE plots) of all Continuous Variables', fontsize=20, y=1.08)
        if verbose == 1:
            plt.show();
    else:
        ######### This is for Classification problems only ########
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
        for k, color2 in zip(range(noplots),colors):
            each_conti = conti[k]
            if dft[each_conti].isnull().sum() > 0:
                dft[each_conti].fillna(0, inplace=True)
            for target_var, color2, class_label in zip(target_vars,colors,classes):
                plt.subplot(rows, cols, k+1)
                ax1 = plt.gca()
                try:
                    if legend_flag <= label_limit:
                        sns.distplot(dft.loc[dft[dep]==target_var][each_conti],
                            hist=False, kde=True,
                        #dft.ix[dft[dep]==target_var][each_conti].hist(
                            bins=binsize, ax= ax1,
                            label=class_label, color=color2)
                        ax1.set_title('Distribution of %s' %each_conti)
                        legend_flag += 1
                    else:
                        sns.distplot(dft.loc[dft[dep]==target_var][each_conti],bins=binsize, ax= ax1,
                        label=class_label, hist=False, kde=True,
                        color=color2)
                        legend_flag += 1
                        ax1.set_title('Normed Histogram of %s' %each_conti)
                except:
                    pass
            ax1.legend(loc='best')
        fig.tight_layout();
        if verbose == 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
        fig.suptitle('Histograms (KDE plots) of all Continuous Variables', fontsize=20, y=1.08)
        ###### Now draw the distribution of the target variable in Classification only ####
        if problem_type.endswith('Classification'):
            if dft[dep].dtype == object:
                dft[dep] = dft[dep].factorize()[0]
            col = 2
            row = 1
            fig, (ax1,ax2) = plt.subplots(row, col)
            fig.set_figheight(5)
            fig.set_figwidth(15)
            fig.suptitle('%s : Distribution of Target Variable' %dep, fontsize=20,y=1.08)
            #fig.subplots_adjust(hspace=0.3) ### This controls the space betwen rows
            #fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
            ###### Precentage Distribution is first #################
            dft[dep].value_counts(1).plot(ax=ax1,kind='bar')
            for p in ax1.patches:
                ax1.annotate(str(round(p.get_height(),2)), (round(p.get_x()*1.01,2), round(p.get_height()*1.01,2)))
            ax1.set_title('Percentage Distribution of Target = %s' %dep, fontsize=10, y=1.05)
            #### Freq Distribution is next ###########################
            dft[dep].value_counts().plot(ax=ax2,kind='bar')
            for p in ax2.patches:
                ax2.annotate(str(round(p.get_height(),2)), (round(p.get_x()*1.01,2), round(p.get_height()*1.01,2)))
            ax2.set_xticks(dft[dep].unique().tolist())
            ax2.set_xticklabels(classes)
            ax2.set_title('Freq Distribution of Target Variable = %s' %dep,  fontsize=10,y=1.05)
        else:
            ############################################################################
            width_size = 5
            height_size = 5
            fig = plt.figure(figsize=(width_size,height_size))
            dft[dep].plot(kind='hist')
            fig.suptitle('%s : Distribution of Target Variable' %dep, fontsize=20,y=1.05)
            fig.tight_layout();
        if verbose == 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
    ####### End of Distplots ###########
    return imgdata_list

##### Standardize all the variables in One step. But be careful !
#### All the variables must be numeric for this to work !!
def draw_violinplot(df, dep, nums,verbose,chart_format, modeltype='Regression'):
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
        for row in range(rows):
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
            fig.suptitle('Violin Plot of all Continuous Variables', fontsize=20,y=1.08)
            if verbose == 1:
                plt.show();
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, image_count, chart_format))
                image_count += 1
    else :
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
            fig.suptitle('Box Plots without Outliers shown',  fontsize=20,y=1.08)
            if verbose == 1:
                plt.show();
            if verbose == 2:
                imgdata_list.append(save_image_data(fig, image_count, chart_format))
                image_count += 1
        #########################################
    return imgdata_list
    ########## End of Violin Plots #########

#### Drawing Date Variables is very important in Time Series data
def draw_date_vars(df,dep,datevars, num_vars,verbose, chart_format, modeltype='Regression'):
    #### Now you want to display 2 variables at a time to see how they change over time
    ### Don't change the number of cols since you will have to change rows formula as well
    imgdata_list = list()
    image_count = 0
    N = len(num_vars)
    if N < 2:
        var1 = num_vars[0]
        width_size = 5
        height_size = 5
        fig = plt.figure(figsize=(width_size,height_size))
        df[var1].plot(title=var1, label=var1)
        fig.suptitle('Time Series Plot of %s' %var1, fontsize=20,y=1.08)
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
        return imgdata_list
    if isinstance(df.index, pd.DatetimeIndex) :
        df =  df[:]
        pass
    else:
        df = df[:]
        try:
            col = datevars[0]
            if df[col].map(lambda x: 0 if len(str(x)) == 4 else 1).sum() == 0:
                if df[col].min() > 1900 or df[col].max() < 2100:
                    df[col] = df[col].map(lambda x: '01-01-'+str(x) if len(str(x)) == 4 else x)
                    df.index = pd.to_datetime(df.pop(col), infer_datetime_format=True)
                else:
                    print('%s could not be indexed. Could not draw date_vars.' %col)
                    return imgdata_list
            else:
                df.index = pd.to_datetime(df.pop(col), infer_datetime_format=True)
        except:
            print('%s could not be indexed. Could not draw date_vars.' %col)
            return imgdata_list
    ####### Draw the time series for Regression and DepVar
    if modeltype == 'Regression' or dep == None or dep == '':
        width_size = 15
        height_size = 4
        image_count = 0
        cols = 2
        combos = combinations(num_vars, 2)
        combs = copy.deepcopy(combos)
        noplots = int((N**2-N)/2)
        rows = int((noplots/cols)+0.99)
        counter = 1
        fig = plt.figure(figsize=(width_size,rows*height_size))
        for (var1,var2) in combos:
            plt.subplot(rows,cols,counter)
            ax1 = plt.gca()
            df[var1].plot(secondary_y=True, label=var1, ax=ax1)
            df[var2].plot(title=var2 +' (left_axis) vs. ' + var1+' (right_axis)', ax=ax1)
            plt.legend(loc='best')
            counter += 1
        fig.suptitle('Time Series Plot by %s: Pairwise Continuous Variables' %col, fontsize=20,y=1.08)
        #fig.tight_layout();
        if verbose == 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
            image_count += 1
    else:
        ######## This is for Classification problems only
        image_count = 0
        classes = df[dep].factorize()[1].tolist()
        classes = copy.deepcopy(classes)
        ##### Now separate out the drawing of time series data by the number of classes ###
        colors = cycle('gkbyrcmgkbyrcmgkbyrcmgkbyr')
        target_vars = df[dep].unique()
        if type(classes[0])==int or type(classes[0])==float:
            classes = [str(x) for x in classes]
        cols = 2
        noplots = int((N**2-N)/2)
        rows = int((noplots/cols)+0.99)
        fig = plt.figure(figsize=(width_size,rows*height_size))
        for target_var, class_label, color2 in zip(target_vars, classes, colors):
            ## Once the date var has been set as the index, you can draw num variables against it
            df_target = df[df[dep]==target_var]
            combos = combinations(num_vars, 2)
            combs = copy.deepcopy(combos)
            counter = 1
            for (var1,var2) in combos :
                plt.subplot(rows,cols,counter)
                ax1 = plt.gca()
                df_target[var1].plot(secondary_y=True, label=var1,ax=ax1)
                df_target[var2].plot(title='Target = '+class_label+': '+var2 +' (left_axis) vs. '+var1,ax=ax1)
                plt.legend(loc='best')
                counter += 1
        fig.suptitle('Time Series Plot by %s: Continuous Variables Pair' %col, fontsize=20,y=1.08)
        if verbose == 1:
            plt.show();
        if verbose == 2:
            imgdata_list.append(save_image_data(fig, image_count, chart_format))
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
                    if verbose == 1:
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
                    if verbose == 1:
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
    date_vars = [x for x in date_vars if x not in cats+bool_vars ]
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

##################
def classify_print_vars(filename,sep, max_rows_analyzed,max_cols_analyzed,
                        depVar='',dfte=None, header=0,verbose=0):
    start_time=time.time()
    if filename == '':
        dft = dfte[:]
        pass
    elif filename != '' and not filename.endswith(('.xls', '.xlsx')):
        codex = ['utf-8', 'iso-8859-11', 'cpl252', 'latin1']
        for code in codex:
            try:
                dfte = pd.read_csv(filename,sep=sep,index_col=None,encoding=code)
                break
            except:
                print('File encoding decoder %s does not work for this file' %code)
                continue
    elif filename != '' and filename.endswith(('xlsx','xls')):
        try:
            dfte = pd.read_excel(filename, header=header)
        except:
            print('Could not load your Excel file')
            return
    else:
        print('Could not read your data file')
        return
    try:
        print('Shape of your Data Set: %s' %(dfte.shape,))
    except:
        print('None of the decoders work...')
        return
    orig_preds = [x for x in list(dfte) if x not in [depVar]]
    #################    CLASSIFY  COLUMNS   HERE    ######################
    var_df = classify_columns(dfte[orig_preds], verbose)
    #####       Classify Columns   ################
    IDcols = var_df['id_vars']
    discrete_string_vars = var_df['nlp_vars']+var_df['discrete_string_vars'] 
    cols_delete = var_df['cols_delete']             
    bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
    categorical_vars = var_df['cat_vars'] + var_df['factor_vars']
    continuous_vars = var_df['continuous_vars']
    date_vars = var_df['date_vars']
    int_vars = var_df['int_vars']
    preds = [x for x in orig_preds if x not in IDcols+cols_delete+discrete_string_vars]
    if len(IDcols+cols_delete+discrete_string_vars) == 0:
        print('    No variables removed since no ID or low-information variables found in data set')
    else:
        print('    %d variables removed since they were ID or low-information variables' 
                                %len(IDcols+cols_delete+discrete_string_vars))
    #############    Sample data if too big and find problem type   #############################
    if dfte.shape[0]>= max_rows_analyzed:
        print('Since Number of Rows in data %d exceeds maximum, randomly sampling %d rows for EDA...' %(len(dfte),max_rows_analyzed))
        dft = dfte.sample(max_rows_analyzed, random_state=0)
    else:
        dft = copy.deepcopy(dfte)
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
        elif dft[depVar].dtype == bool:
            classes =  dft[depVar].unique().astype(int).tolist()
        elif dft[depVar1].dtype == float and problem_type.endswith('Classification'):
            classes = dft[depVar1].factorize()[1].tolist()
        else:
            classes = []
    #############  Check if there are too many columns to visualize  ################
    if len(continuous_vars) >= max_cols_analyzed:
        #########     In that case, SELECT IMPORTANT FEATURES HERE   ######################
        if problem_type.endswith('Classification') or problem_type == 'Regression':
            print('%d numeric variables in data exceeds limit, taking top %d variables' %(len(
                                            continuous_vars), max_cols_analyzed))
            important_features,num_vars = find_top_features_xgb(dft,preds,continuous_vars,
                                                         depVar,problem_type,verbose)
            if len(important_features) >= max_cols_analyzed:
                ### Limit the number of features to max columns analyzed ########
                important_features = important_features[:max_cols_analyzed]
            dft = dft[important_features+[depVar]]
            #### Time to  classify the important columns again ###
            var_df = classify_columns(dft[important_features], verbose)
            IDcols = var_df['id_vars']
            discrete_string_vars = var_df['nlp_vars']+var_df['discrete_string_vars'] 
            cols_delete = var_df['cols_delete']             
            bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
            categorical_vars = var_df['cat_vars'] + var_df['factor_vars']
            continuous_vars = var_df['continuous_vars']
            date_vars = var_df['date_vars']
            int_vars = var_df['int_vars']
            preds = [x for x in important_features if x not in IDcols+cols_delete+discrete_string_vars]
            if len(IDcols+cols_delete+discrete_string_vars) == 0:
                print('    No variables removed since no ID or low-information variables found in data')
            else:
                print('    %d variables removed since they were ID or low-information variables' 
                                        %len(IDcols+cols_delete+discrete_string_vars))
            dft = dft[preds+[depVar]]
        else:
            continuous_vars = continuous_vars[:max_cols_analyzed]
            print('%d numeric variables in data exceeds limit, taking top %d variables' %(len(
                                            continuous_vars, max_cols_analyzed)))
    elif len(continuous_vars) < 1:
        print('No continuous variables in this data set. No visualization can be performed')
        ### Return data frame as is #####
        return dfte
    else:
        #########     If above 1 but below limit, leave features as it is   ######################
        if depVar != '':
            dft = dft[preds+[depVar]]
        else:
            dft = dft[preds]
    #############   N
    ppt = pprint.PrettyPrinter(indent=4)
    if verbose==1 and len(cols_list) <= max_cols_analyzed:
        marthas_columns(dft,verbose)
        print("   Columns to delete:")
        ppt.pprint('   %s' % cols_delete)
        print("   Boolean variables %s ")
        ppt.pprint('   %s' % bool_vars)
        print("   Categorical variables %s ")
        ppt.pprint('   %s' % categorical_vars)
        print("   Continuous variables %s " )
        ppt.pprint('   %s' % continuous_vars)
        print("   Discrete string variables %s " )
        ppt.pprint('   %s' % discrete_string_vars)
        print("   Date and time variables %s " )
        ppt.pprint('   %s' % date_vars)
        print("   ID variables %s ")
        ppt.pprint('   %s' % IDcols)
        print("   Target variable %s ")
        ppt.pprint('   %s' % depVar)
    elif verbose==1 and len(cols_list) > max_cols_analyzed:
        print('   Total columns > %d, too numerous to list.' %max_cols_analyzed)
    ###################   Time to drop the columns to be deleted #############
    #if cols_delete != []:
    #    dft.drop(cols_delete, axis=1, inplace=True)
    #print('Time to run Print_Classify_Variables (in seconds) = %0.3f' %(time.time()-start_time))
    return dft,depVar,IDcols,bool_vars,categorical_vars,continuous_vars,discrete_string_vars,date_vars,classes,problem_type
####################################################################
def marthas_columns(data,verbose=0):
    """
    This program is named  in honor of my one of students who came up with the idea for it.
    It's a neat way of looking at data compared to the boring describe() function in Pandas.
    """
    data = data[:]
    print('Data Set Shape: %d rows, %d cols\n' % data.shape)
    if data.shape[1] > 25:
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
            print('\n------\n')
################################################
######### NEW And FAST WAY to CLASSIFY COLUMNS IN A DATA SET #######
def classify_columns(df_preds, verbose=0):
    """
    Takes a dataframe containing only predictors to be classified into various types.
    DO NOT SEND IN A TARGET COLUMN since it will try to include that into various columns.
    Returns a data frame containing columns and the class it belongs to such as numeric,
    categorical, date or id column, boolean, nlp, discrete_string and cols to delete...
    ####### Returns a dictionary with 10 kinds of vars like the following: # continuous_vars,int_vars
    # cat_vars,factor_vars, bool_vars,discrete_string_vars,nlp_vars,date_vars,id_vars,cols_delete
    """
    print('Classifying variables in data set...')
    #### Cat_Limit defines the max number of categories a column can have to be called a categorical colum 
    cat_limit = 15
    def add(a,b):
        return a+b
    train = df_preds[:]
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
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [
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
    if len(var_df.loc[discrete_or_nlp==1]) != 0:
        for col in discrete_or_nlp_vars:
            #### first fill empty or missing vals since it will blowup ###
            train[col] = train[col].fillna('  ')
            if train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= 50 and len(train[col].value_counts()
                        ) < len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) < len(train) and col not in string_bool_vars:
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
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in ['int8','int16',
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
    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    ## This is an EXTREMELY complicated logic for cat vars. Don't change it unless you test it many times!
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= cat_limit and len(train[col].value_counts()) != len(train):
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if col not in num_bool_vars:
                    var_df.loc[var_df['index']==col,'numeric'] = 1
    cat_vars = list(var_df[(var_df['cat'] ==1)]['index'])
    continuous_vars = list(var_df[(var_df['numeric'] ==1)]['index'])
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['continuous_vars'] = continuous_vars
    ###### This is where you consoldate the numbers ###########
    var_dict_sum = dict(zip(var_df.values[:,0], var_df.values[:,2:].sum(1)))
    for col, sumval in var_dict_sum.items():
        if sumval == 0:
            print('%s of type=%s is not classified' %(col,train[col].dtype))
        elif sumval > 1:
            print('%s of type=%s is classified into more then one type' %(col,train[col].dtype))
        else:
            pass
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
    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()])
    if len_sum_all_cols == orig_cols_total:
        print('    %d Predictors classified...' %orig_cols_total)
        print('        This does not include the Target column(s)')
    else:
        print('No of columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]
        print('    Missing columns = %s' %set(list(train))-set(flat_list))
    return sum_all_cols
#################################################################################
from collections import Counter
import time
from collections import OrderedDict
def remove_variables_using_fast_correlation(df,numvars,corr_limit = 0.70,verbose=0):
    """
    Removes variables that are highly correlated using a pair-wise
    high-correlation knockout method. It is highly efficient and hence can work on thousands
    of variables in less than a minute, even on a laptop. Only send in a list of numeric
    variables, otherwise, it will blow-up!
    Correlation = 0.70 This is the highest correlation that any two variables can have.
    Above this, and one of them gets knocked out: this is decided in the shootout stage
    after the initial round of cutoffs for pair-wise correlations...It returns a list of
    clean variables that are uncorrelated (atleast in a pair-wise sense).
    """
    flatten = lambda l: [item for sublist in l for item in sublist]
    flatten_items = lambda dic: [x for x in dic.items()]
    flatten_keys = lambda dic: [x for x in dic.keys()]
    flatten_values = lambda dic: [x for x in dic.values()]
    start_time = time.time()
    print('Number of numeric variables = %d' %len(numvars))
    corr_pair_count_dict, rem_col_list, temp_corr_list,correlated_pair_dict  = find_corr_vars(df[numvars].corr())
    temp_dict = Counter(flatten(flatten_items(correlated_pair_dict)))
    temp_corr_list = []
    for name, count in temp_dict.items():
        if count >= 2:
            temp_corr_list.append(name)
    temp_uncorr_list = []
    for name, count in temp_dict.items():
        if count == 1:
            temp_uncorr_list.append(name)
    ### Do another correlation test to remove those that are correlated to each other ####
    corr_pair_count_dict2, rem_col_list2 , temp_corr_list2, correlated_pair_dict2 = find_corr_vars(
                            df[rem_col_list+temp_uncorr_list].corr(),corr_limit)
    final_dict = Counter(flatten(flatten_items(correlated_pair_dict2)))
    #### Make sure that these lists are sorted and compared. Otherwise, you will get False compares.
    if temp_corr_list2.sort() == temp_uncorr_list.sort():
        ### if what you sent in, you got back the same, then you now need to pick just one: 
        ###   either keys or values of this correlated_pair_dictionary. Which one to pick?
        ###   Here we select the one which has the least overall correlation to rem_col_list
        ####  The reason we choose overall mean rather than absolute mean is the same reason in finance
        ####   A portfolio that has lower overall mean is better than  a portfolio with higher correlation
        corr_keys_mean = df[rem_col_list+flatten_keys(correlated_pair_dict2)].corr().mean().mean()
        corr_values_mean = df[rem_col_list+flatten_values(correlated_pair_dict2)].corr().mean().mean()
        if corr_keys_mean <= corr_values_mean: 
            final_uncorr_list = flatten_keys(correlated_pair_dict2)
        else:
            final_uncorr_list = flatten_values(correlated_pair_dict2)
    else:
        final_corr_list = []
        for name, count in final_dict.items():
            if count >= 2:
                final_corr_list.append(name)
        final_uncorr_list = []
        for name, count in final_dict.items():
            if count == 1:
                final_uncorr_list.append(name)
    ####  Once we have chosen a few from the highest corr list, we add them to the highest uncorr list#####
    selected = copy.deepcopy(final_uncorr_list)
    #####  Now we have reduced the list of vars and these are ready to be used ####
    final_list = list(OrderedDict.fromkeys(selected + rem_col_list))
    if int(len(numvars)-len(final_list)) == 0:
              print('    No variables were removed since no highly correlated variables found in data')
    else:
        print('    Number of variables removed due to high correlation = %d ' %(len(numvars)-len(final_list)))
    #print('    Time taken for removing highly correlated variables (in secs)=%0.0f' %(time.time()-start_time))
    return final_list

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
################      Find top features using XGB     ###################
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.multioutput import MultiOutputClassifier
import copy
from sklearn.multiclass import OneVsRestClassifier
from collections import OrderedDict
def find_top_features_xgb(train,preds,numvars,target,modeltype,corr_limit,verbose=0):
    """
    This is a fast utility that uses XGB to find top features. You
    It returns a list of important features.
    Since it is XGB, you dont have to restrict the input to just numeric vars.
    You can send in all kinds of vars and it will take care of transforming it. Sweet!
    """
    subsample =  0.5
    col_sub_sample = 0.5
    train = copy.deepcopy(train)
    start_time = time.time()
    test_size = 0.2
    seed = 1
    n_splits = 5
    kf = KFold(n_splits=n_splits,random_state= 33)
    rem_vars = left_subtract(preds,numvars)
    if len(numvars) > 0:
        final_list = remove_variables_using_fast_correlation(train,numvars,corr_limit,verbose)
    else:
        final_list = numvars[:]
    print('    Adding %s categorical variables to reduced numeric variables  of %d' %(
                            len(rem_vars),len(final_list)))
    preds = final_list+rem_vars
    ########    Drop Missing value rows since XGB for some reason  #########
    ########    can't handle missing values in early stopping rounds #######
    train.dropna(axis=0,subset=preds+[target],inplace=True)
    ########   Dont move this train and y definition anywhere else ########
    y = train[target]
    ######################################################################
    important_features = []
    if modeltype == 'Regression':
        model_xgb = XGBRegressor(objective='reg:linear', n_estimators=100,subsample=subsample,
                                colsample_bytree=col_sub_sample,reg_alpha=0.5, reg_lambda=0.5, 
                                 seed=1,n_jobs=-1,random_state=1)
        eval_metric = 'rmse'
    else:
        #### This is for Classifiers only
        classes = np.unique(train[target].values)
        if len(classes) == 2:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                colsample_bytree=col_sub_sample,gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=5, min_child_weight=1, missing=-999, n_estimators=100,
                n_jobs=-1, nthread=None, objective='binary:logistic',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5, scale_pos_weight=1,
                seed=1, silent=True)
            eval_metric = 'logloss'
        else:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                        colsample_bytree=col_sub_sample, gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=5, min_child_weight=1, missing=-999, n_estimators=100,
                n_jobs=-1, nthread=None, objective='multi:softmax',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5, scale_pos_weight=1,
                seed=1, silent=True)
            eval_metric = 'mlogloss'
    ####   This is where you start to Iterate on Finding Important Features ################
    train_p = train[preds]
    if train_p.shape[1] < 10:
        iter_limit = 2
    else:
        iter_limit = int(train_p.shape[1]/5+0.5)
    print('Selected No. of variables = %d ' %(train_p.shape[1],))
    print('Finding Important Features...')
    for i in range(0,train_p.shape[1],iter_limit):
        if verbose == 1:
            print('        in %d variables' %(train_p.shape[1]-i))
        if train_p.shape[1]-i < iter_limit:
            X = train_p.iloc[:,i:]
            if modeltype == 'Regression':
                train_part = int((1-test_size)*X.shape[0])
                X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
            else:
                X_train, X_cv, y_train, y_cv = train_test_split(X, y, 
                                                            test_size=test_size, random_state=seed)
            try:
                model_xgb.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_cv,y_cv)],
                                    eval_metric=eval_metric,verbose=False)
            except:
                print('XGB is Erroring. Check if there are missing values in your data and try again...')
                return [], []
            try:
                [important_features.append(x) for x in list(pd.concat([pd.Series(model_xgb.feature_importances_
                        ),pd.Series(list(X_train.columns.values))],axis=1).rename(columns={0:'importance',1:'column'
                    }).sort_values(by='importance',ascending=False)[:25]['column'])]
            except:
                print('Model training error in find top feature...')
                important_features = copy.deepcopy(preds)
                return important_features, [], []
        else:
            X = train_p[list(train_p.columns.values)[i:train_p.shape[1]]]
            #### Split here into train and test #####            
            if modeltype == 'Regression':
                train_part = int((1-test_size)*X.shape[0])
                X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
            else:
                X_train, X_cv, y_train, y_cv = train_test_split(X, y, 
                                                            test_size=test_size, random_state=seed)
            model_xgb.fit(X_train,y_train,early_stopping_rounds=5,
                          eval_set=[(X_cv,y_cv)],eval_metric=eval_metric,verbose=False)
            try:
                [important_features.append(x) for x in list(pd.concat([pd.Series(model_xgb.feature_importances_
                        ),pd.Series(list(X_train.columns.values))],axis=1).rename(columns={0:'importance',1:'column'
                    }).sort_values(by='importance',ascending=False)[:25]['column'])]
                important_features = list(OrderedDict.fromkeys(important_features))
            except:
                print('Multi Label possibly no feature importances.')
                important_features = copy.deepcopy(preds)
    important_features = list(OrderedDict.fromkeys(important_features))
    print('    Found %d important features' %len(important_features))
    #print('    Time taken (in seconds) = %0.0f' %(time.time()-start_time))
    numvars = [x for x in numvars if x in important_features]
    return important_features, numvars
###############################################

#################################################################################
if __name__ == "__main__":
    print("""AutoViz_Class is imported. Call AutoViz(filename, sep=',', depVar='', dfte=None, header=0, verbose=0,
                            lowess=False,chart_format='svg',max_rows_analyzed=150000,max_cols_analyzed=30)""")
else:
    print("""Imported AutoViz_Class. Call by using AutoViz(filename, sep=',', depVar='', dfte=None, header=0, verbose=0,
                            lowess=False,chart_format='svg',max_rows_analyzed=150000,max_cols_analyzed=30)""")
