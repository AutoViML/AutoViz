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
##########################################################################################
from autoviz.AutoViz_Holo import AutoViz_Holo
from autoviz.AutoViz_Utils import *
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
        ###########            Version V0.0.68 1/10/20                          ######
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
        self.wordcloud = {
        'name': 'wordcloud',
        'heading': 'Word Cloud Plots of NLP or String vars',
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
            #print("Nothing to add Plot not being added")
            pass
        else:
            getattr(self, plotname)["plots"].append(X)

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
            getattr(self,plotname)["subheading"].append(X)

    def AutoViz(self, filename, sep=',', depVar='', dfte=None, header=0, verbose=0,
                            lowess=False,chart_format='svg',max_rows_analyzed=150000,
                                max_cols_analyzed=30, save_plot_dir=None):
        """
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
        ####################################################################################
        if chart_format.lower() in ['bokeh','server','bokeh_server','bokeh-server', 'html']:
            dft = AutoViz_Holo(filename, sep, depVar, dfte, header, verbose,
                        lowess,chart_format,max_rows_analyzed,
                            max_cols_analyzed, save_plot_dir)
        else:
            dft = self.AutoViz_Main(filename, sep, depVar, dfte, header, verbose,
                        lowess,chart_format,max_rows_analyzed,
                            max_cols_analyzed, save_plot_dir)
        return dft
    
    def AutoViz_Main(self, filename, sep=',', depVar='', dfte=None, header=0, verbose=0,
                            lowess=False,chart_format='svg',max_rows_analyzed=150000,
                                max_cols_analyzed=30, save_plot_dir=None):
        """
        ##############################################################################
        ##### AUTOVIZ_MAIN PERFORMS AUTO VISUALIZATION OF ANY DATA USING MATPLOTLIB ##
        ##############################################################################
        """
        corr_limit = 0.7  ### This is needed to remove variables correlated above this limit
        ######### create a directory to save all plots generated by autoviz ############
        ############    THis is where you save the figures in a target directory #######
        target_dir = 'AutoViz'
        if not depVar is None:
            if depVar != '':
                target_dir = copy.deepcopy(depVar)
        if save_plot_dir is None:
            mk_dir = os.path.join(".","AutoViz_Plots")
        else:
            mk_dir = copy.deepcopy(save_plot_dir)
        if verbose == 2 and not os.path.isdir(mk_dir):
            os.mkdir(mk_dir)
        mk_dir = os.path.join(mk_dir,target_dir)
        if verbose == 2 and not os.path.isdir(mk_dir):
            os.mkdir(mk_dir)
        ############   Start the clock here and classify variables in data set first ########
        start_time = time.time()
        try:
            dft, depVar,IDcols,bool_vars,cats,continuous_vars,discrete_string_vars,date_vars,classes,problem_type,selected_cols = classify_print_vars(
                                                filename,sep,max_rows_analyzed, max_cols_analyzed,
                                                depVar,dfte,header,verbose)
        except:
            print('Not able to read or load file. Please check your inputs and try again...')
            return None
        ##### This is where we start plotting different kinds of charts depending on dependent variables
        if depVar == None or depVar == '':
         ##### This is when No dependent Variable is given #######
            try:
                svg_data = draw_pair_scatters(dft,continuous_vars,problem_type,verbose,chart_format,
                                                depVar,classes,lowess, mk_dir)
                self.add_plots('pair_scatter',svg_data)
            except Exception as e:
                print(e)
                print('Could not draw Pair Scatter Plots')
            try:
                svg_data = draw_distplot(dft, bool_vars+cats, continuous_vars,verbose,chart_format,problem_type,
                                    depVar,classes, mk_dir)
                self.add_plots('dist_plot',svg_data)
            except:
                print('Could not draw Distribution Plot')
            try:
                svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                self.add_plots('violin_plot',svg_data)
            except:
                print('Could not draw Violin Plot')
            try:
                #### Since there is no depependent variable in this dataset, you can leave it as-is
                numeric_cols = dft.select_dtypes(include='number').columns.tolist()
                numeric_cols = list_difference(numeric_cols, date_vars)
                svg_data = draw_heatmap(dft, numeric_cols, verbose,chart_format, date_vars, depVar,
                                    problem_type,classes, mk_dir)
                self.add_plots('heat_map',svg_data)
            except:
                print('Could not draw Heat Map')
            if date_vars != [] and len(continuous_vars) > 0:
                try:
                    svg_data = draw_date_vars(dft,depVar,date_vars,
                                              continuous_vars,verbose,chart_format,problem_type, mk_dir)
                    self.add_plots('date_plot',svg_data)
                except:
                    print('Could not draw Date Vars')
            if len(continuous_vars) > 0:
                try:
                    svg_data = draw_barplots(dft,cats,continuous_vars, problem_type,
                                    verbose,chart_format,depVar,classes, mk_dir)
                    self.add_plots('bar_plot',svg_data)
                except:
                    print('Could not draw Bar Plots')
            else:
                print ('No Continuous Variables at all in this dataset...')
        else:
            if problem_type=='Regression':
                ############## This is a Regression Problem #################
                try:
                    svg_data = draw_scatters(dft,
                                    continuous_vars,verbose,chart_format,problem_type,depVar,classes,lowess, mk_dir)
                    self.add_plots('scatter_plot',svg_data)
                except Exception as e:
                    print("Exception Drawing Scatter Plots")
                    print(e)
                    traceback.print_exc()
                    print('Could not draw Scatter Plots')
                try:
                    svg_data = draw_pair_scatters(dft,continuous_vars,problem_type,verbose,chart_format,
                                                    depVar,classes,lowess, mk_dir)
                    self.add_plots('pair_scatter',svg_data)
                except:
                    print('Could not draw Pair Scatter Plots')
                try:
                    if type(depVar) == str:
                        othernums = [x for x in continuous_vars if x not in [depVar]]
                    else:
                        othernums = [x for x in continuous_vars if x not in depVar]
                    if len(othernums) >= 1:
                        svg_data = draw_distplot(dft, bool_vars+cats, continuous_vars,verbose,chart_format,
                                            problem_type, depVar, classes, mk_dir)
                        self.add_plots('dist_plot',svg_data)
                    else:
                        print('No continuous var in data set: hence no distribution plots')
                except:
                    print('Could not draw Distribution Plots')
                try:
                    svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                    self.add_plots('violin_plot',svg_data)
                except:
                    print('Could not draw Violin Plots')
                try:
                    numeric_cols = [x for x in dft.select_dtypes(include='number').columns.tolist() if x not in [depVar]]
                    numeric_cols = list_difference(numeric_cols, date_vars)
                    svg_data = draw_heatmap(dft,
                                        numeric_cols, verbose,chart_format, date_vars, depVar,
                                            problem_type, classes, mk_dir)
                    self.add_plots('heat_map',svg_data)
                except:
                    print('Could not draw Heat Maps')
                if date_vars != [] and len(continuous_vars) > 0:
                    try:
                        svg_data = draw_date_vars(
                            dft,depVar,date_vars,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                        self.add_plots('date_plot',svg_data)
                    except:
                        print('Could not draw Time Series plots')
                if len(continuous_vars) > 0:
                    try:
                        svg_data = draw_pivot_tables(dft,find_remove_duplicates(cats+bool_vars),
                                    continuous_vars,problem_type,verbose,chart_format,depVar,classes, mk_dir)
                        self.add_plots('pivot_plot',svg_data)
                    except:
                        print('Could not draw Pivot Charts against Dependent Variable')
                    try:
                        svg_data = draw_barplots(dft,cats,continuous_vars,problem_type,verbose,
                                                    chart_format,depVar,classes, mk_dir)
                        self.add_plots('bar_plot',svg_data)
                        #self.add_plots('bar_plot',None)
                    except:
                        print('Could not draw Bar Charts')
                else:
                    print ('No Continuous Variables at all in this dataset...')
                if verbose <= 1:
                    print('All Plots done')
                else:
                    print('All Plots are saved in %s' %mk_dir)
            else :
                ############ This is a Classification Problem ##################
                try:
                    svg_data = draw_scatters(dft,continuous_vars,
                                             verbose,chart_format,problem_type,depVar, classes,lowess, mk_dir)
                    self.add_plots('scatter_plot',svg_data)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print("Exception Drawing Scatter Plots")
                    print('Could not draw Scatter Plots')
                try:
                    svg_data = draw_pair_scatters(dft,continuous_vars,
                                                  problem_type,verbose,chart_format,depVar,classes,lowess, mk_dir)
                    self.add_plots('pair_scatter',svg_data)
                except:
                    print('Could not draw Pair Scatter Plots')
                try:
                    if type(depVar) == str:
                        othernums = [x for x in continuous_vars if x not in [depVar]]
                    else:
                        othernums = [x for x in continuous_vars if x not in depVar]
                    if len(othernums) >= 1:
                        svg_data = draw_distplot(dft, bool_vars+cats, continuous_vars,verbose,chart_format,
                                                problem_type,depVar,classes, mk_dir)
                        self.add_plots('dist_plot',svg_data)
                    else:
                        print('No continuous var in data set: hence no distribution plots')
                except:
                    print('Could not draw Distribution Plots')
                try:
                    svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                    self.add_plots('violin_plot',svg_data)
                except:
                    print('Could not draw Violin Plots')
                try:
                    numeric_cols = [x for x in dft.select_dtypes(include='number').columns.tolist() if x not in [depVar]]
                    numeric_cols = list_difference(numeric_cols, date_vars)
                    svg_data = draw_heatmap(dft, numeric_cols,
                                            verbose,chart_format, date_vars, depVar,problem_type,
                                            classes, mk_dir)
                    self.add_plots('heat_map',svg_data)
                except:
                    print('Could not draw Heat Maps')
                if date_vars != [] and len(continuous_vars) > 0:
                    try:
                        svg_data = draw_date_vars(dft,depVar,date_vars,
                                                  continuous_vars,verbose,chart_format,problem_type, mk_dir)
                        self.add_plots('date_plot',svg_data)
                    except:
                        print('Could not draw Time Series plots')
                if len(continuous_vars) > 0:
                    try:
                        svg_data = draw_pivot_tables(
                            dft,find_remove_duplicates(cats+bool_vars),continuous_vars,problem_type,verbose,chart_format,depVar,classes, mk_dir)
                        self.add_plots('pivot_plot',svg_data)
                    except:
                        print('Could not draw Pivot Charts against Dependent Variable')
                    try:
                        svg_data = draw_barplots(dft,find_remove_duplicates(cats+bool_vars),continuous_vars,problem_type,
                                        verbose,chart_format,depVar, classes, mk_dir)
                        self.add_plots('bar_plot',svg_data)
                        pass
                    except:
                        if verbose <= 1:
                            print('Could not draw Bar Charts')
                        pass
                else:
                    print ('No Continuous Variables at all in this dataset...')
        ###### Now you can check for NLP vars or discrete_string_vars to do wordcloud #######
        if len(discrete_string_vars) > 0:
            plotname = 'wordcloud'
            for each_string_var in discrete_string_vars:
                try:
                    svg_data = draw_wordcloud_from_dataframe(dft, each_string_var, chart_format, plotname, 
                                    depVar, mk_dir, verbose=0)
                    self.add_plots(plotname,svg_data)
                except:
                    print('Could not draw wordcloud plot for %s' %each_string_var)
        ### Now print the time taken to run charts for AutoViz #############
        print('Time to run AutoViz = %0.0f seconds ' %(time.time()-start_time))
        if verbose <= 1:
            print ('\n ###################### AUTO VISUALIZATION Completed ########################')
        return dft
#############################################################################################
