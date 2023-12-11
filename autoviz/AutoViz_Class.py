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
sns.set(style="ticks", color_codes=True)
import re
import pdb
import pprint
import matplotlib
matplotlib.style.use('seaborn-v0_8-ticks')
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
from autoviz.AutoViz_Utils import save_image_data, save_html_data, analyze_problem_type, draw_pivot_tables, draw_scatters
from autoviz.AutoViz_Utils import draw_pair_scatters, plot_fast_average_num_by_cat, draw_barplots, draw_heatmap
from autoviz.AutoViz_Utils import draw_distplot, draw_violinplot, draw_date_vars, catscatter, draw_catscatterplots
from autoviz.AutoViz_Utils import list_difference, search_for_word_in_list, analyze_ID_columns, start_classifying_vars
from autoviz.AutoViz_Utils import analyze_columns_in_dataset, find_remove_duplicates, load_file_dataframe, classify_print_vars
from autoviz.AutoViz_Utils import marthas_columns, EDA_find_remove_columns_with_infinity, return_dictionary_list
from autoviz.AutoViz_Utils import remove_variables_using_fast_correlation, count_freq_in_list, find_corr_vars, left_subtract
from autoviz.AutoViz_Utils import convert_train_test_cat_col_to_numeric, return_factorized_dict, convert_a_column_to_numeric
from autoviz.AutoViz_Utils import convert_all_object_columns_to_numeric, find_top_features_xgb, convert_a_mixed_object_column_to_numeric
from autoviz.AutoViz_NLP import draw_word_clouds
#############################################################################################
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
        self.catscatter_plot = {
        'name': 'catscatter',
        'heading': 'Cat-Scatter  Plots of categorical vars',
        'plots': [],
        'subheading':[],#"\n".join(subheading)
        'desc': [] #"\n".join(desc)
        }  ######## This is for description and images for catscatter plots ###


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

    def AutoViz(self, filename, sep=',', depVar='', dfte=None, header=0, verbose=1,
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
        if isinstance(depVar, list):
            print('Since AutoViz cannot visualize multi-label targets, choosing first item in targets: %s' %depVar[0])
            depVar = depVar[0]
        
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
        corr_limit = 0.8  ### This is needed to remove variables correlated above this limit
        ######### create a directory to save all plots generated by autoviz ############
        ############    THis is where you save the figures in a target directory #######
        
        if not depVar is None:
            if isinstance(depVar, list):
                target_dir = depVar[0]
            elif isinstance(depVar, str):
                if depVar == '':
                    target_dir = 'AutoViz'
                else:
                    target_dir = copy.deepcopy(depVar)
        else:
            target_dir = 'AutoViz'
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
        ###########  This is where perform data quality checks on data ################
        if verbose >= 1:
            print('To fix data quality issues automatically, import FixDQ from autoviz...')
            data_cleaning_suggestions(dft, target=depVar)

        ##### This is where we start plotting different kinds of charts depending on dependent variables
        if depVar == None or depVar == '':
         ##### This is when No dependent Variable is given #######
            if len(continuous_vars) > 1:
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
                if len(continuous_vars) > 0:
                    svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                    self.add_plots('violin_plot',svg_data)
                else:
                    svg_data = draw_pivot_tables(dft, problem_type, verbose,
                        chart_format,depVar,classes, mk_dir)
                    self.add_plots('pivot_plot',svg_data)
            except:
                print('Could not draw Distribution Plots')
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
            if len(continuous_vars) > 0 and len(cats) > 0:
                try:
                    svg_data = draw_barplots(dft,cats,continuous_vars, problem_type,
                                    verbose,chart_format,depVar,classes, mk_dir)
                    self.add_plots('bar_plot',svg_data)
                except:
                    print('Could not draw Bar Plots')
            else:
                if len(cats) > 1:
                    try:
                        svg_data = draw_catscatterplots(dft,cats, problem_type, verbose, 
                                    chart_format, mk_dir=None)
                        self.add_plots('catscatter_plot',svg_data)
                    except:
                        print ('Could not draw catscatter plots...')
        else:
            if problem_type=='Regression':
                ############## This is a Regression Problem #################
                if len(continuous_vars) > 0:
                    try:
                        svg_data = draw_scatters(dft,
                                        continuous_vars,verbose,chart_format,problem_type,depVar,classes,lowess, mk_dir)
                        self.add_plots('scatter_plot',svg_data)
                    except Exception as e:
                        print("Exception Drawing Scatter Plots")
                        print(e)
                        traceback.print_exc()
                        print('Could not draw Scatter Plots')
                if len(continuous_vars) > 1:
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
                except:
                    print('Could not draw some Distribution Plots')
                try:
                    if len(continuous_vars) > 0:
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
                    print('Could not draw some Heat Maps')
                if date_vars != [] and len(continuous_vars) > 0:
                    try:
                        svg_data = draw_date_vars(
                            dft,depVar,date_vars,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                        self.add_plots('date_plot',svg_data)
                    except:
                        print('Could not draw some Time Series plots')
                if len(cats) > 0 and len(continuous_vars) == 0:
                    ### This is somewhat duplicative with distplot (above) - hence do it only minimally!
                    try:
                        svg_data = draw_pivot_tables(dft, problem_type, verbose,
                            chart_format,depVar,classes, mk_dir)
                        self.add_plots('pivot_plot',svg_data)
                    except:
                        print('Could not draw some Pivot Charts against Dependent Variable')
                if len(continuous_vars) > 0 and len(cats) > 0:
                    try:
                        svg_data = draw_barplots(dft, find_remove_duplicates(cats+bool_vars),continuous_vars,
                                                    problem_type, verbose,chart_format,depVar,classes, mk_dir)
                        self.add_plots('bar_plot',svg_data)
                        #self.add_plots('bar_plot',None)
                    except:
                        print('Could not draw some Bar Charts')
                else:
                    if len(cats) > 1:
                        try:
                            svg_data = draw_catscatterplots(dft,cats, problem_type, verbose, 
                                        chart_format, mk_dir=None)
                            self.add_plots('catscatter_plot',svg_data)
                        except:
                            print ('Could not draw catscatter plots...')
            else :
                ############ This is a Classification Problem ##################
                if len(continuous_vars) > 0:
                    try:
                        svg_data = draw_scatters(dft,continuous_vars,
                                                 verbose,chart_format,problem_type,depVar, classes,lowess, mk_dir)
                        self.add_plots('scatter_plot',svg_data)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        print('Could not draw some Scatter Plots')
                if len(continuous_vars) > 1:
                    try:
                        svg_data = draw_pair_scatters(dft,continuous_vars,
                                                      problem_type,verbose,chart_format,depVar,classes,lowess, mk_dir)
                        self.add_plots('pair_scatter',svg_data)
                    except:
                        print('Could not draw some Pair Scatter Plots')
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
                        print('No continuous var in data set: drawing categorical distribution plots')
                except:
                    print('Could not draw some Distribution Plots')
                try:
                    if len(continuous_vars) > 0:
                        svg_data = draw_violinplot(dft,depVar,continuous_vars,verbose,chart_format,problem_type, mk_dir)
                        self.add_plots('violin_plot',svg_data)
                except:
                    print('Could not draw some Violin Plots')
                try:
                    numeric_cols = [x for x in dft.select_dtypes(include='number').columns.tolist() if x not in [depVar]]
                    numeric_cols = list_difference(numeric_cols, date_vars)
                    svg_data = draw_heatmap(dft, numeric_cols,
                                            verbose,chart_format, date_vars, depVar,problem_type,
                                            classes, mk_dir)
                    self.add_plots('heat_map',svg_data)
                except:
                    print('Could not draw some Heat Maps')
                if date_vars != [] and len(continuous_vars) > 0:
                    try:
                        svg_data = draw_date_vars(dft,depVar,date_vars,
                                                  continuous_vars,verbose,chart_format,problem_type, mk_dir)
                        self.add_plots('date_plot',svg_data)
                    except:
                        print('Could not draw some Time Series plots')
                if len(cats) > 0 and len(continuous_vars) == 0:
                    ### This is somewhat duplicative with distplot (above) - hence do it only minimally!
                    try:
                        svg_data = draw_pivot_tables(dft, problem_type, verbose,
                                        chart_format,depVar,classes, mk_dir)
                        self.add_plots('pivot_plot',svg_data)
                    except:
                        print('Could not draw some Pivot Charts against Dependent Variable')
                if len(continuous_vars) > 0 and len(cats) > 0:
                    try:
                        svg_data = draw_barplots(dft,find_remove_duplicates(cats+bool_vars),continuous_vars,problem_type,
                                        verbose,chart_format, depVar, classes, mk_dir)
                        self.add_plots('bar_plot',svg_data)
                        pass
                    except:
                        if verbose <= 1:
                            print('Could not draw some Bar Charts')
                        pass
                else:
                    if len(cats) > 1:
                        try:
                            svg_data = draw_catscatterplots(dft,cats, problem_type, verbose, 
                                        chart_format, mk_dir=None)
                            self.add_plots('catscatter_plot',svg_data)
                        except:
                            print ('Could not draw catscatter plots...')
        ###### Now you can check for NLP vars or discrete_string_vars to do wordcloud #######
        if len(discrete_string_vars) > 0:
            plotname = 'wordcloud'
            import nltk
            nltk.download('popular')
            for each_string_var in discrete_string_vars:
                try:
                    svg_data = draw_word_clouds(dft, each_string_var, chart_format, plotname, 
                                    depVar, problem_type, classes, mk_dir, verbose=0)
                    self.add_plots(plotname,svg_data)
                except:
                    print('Could not draw wordcloud plot for %s' %each_string_var)
        ### Now print the time taken to run charts for AutoViz #############
        if verbose <= 1:
            print('All Plots done')
        else:
            print('All Plots are saved in %s' %mk_dir)
        print('Time to run AutoViz = %0.0f seconds ' %(time.time()-start_time))
        if verbose <= 1:
            print ('\n ###################### AUTO VISUALIZATION Completed ########################')
        return dft
#############################################################################################
from pandas_dq import Fix_DQ
# Create a new class FixDQ by inheriting from Fix_DQ
class FixDQ(Fix_DQ):
    """
    FixDQ is a great way to clean an entire train data set and apply the same steps in 
    an MLOps pipeline to a test dataset. FixDQ can be used to detect most issues in 
    your data (similar to data_cleaning_suggestions but without the `target` 
    related issues) in one step. Then it fixes those issues it finds during the 
    `fit` method by the `transform` method. This transformer can then be saved 
    (or "pickled") for applying the same steps on test data either at the same 
    time or later.

    FixDQ will perform following data quality cleaning steps:
        It removes ID columns from further processing
        It removes zero-variance columns from further processing
        It identifies rare categories and groups them into a single category 
                    called "Rare"
        It finds infinite values and replaces them with an upper bound based on 
                    Inter Quartile Range
        It detects mixed data types and drops those mixed-type columns from 
                    further processing
        It detects outliers and suggests to remove them or use robust statistics.
        It detects high cardinality features but leaves them as it is.
        It detects highly correlated features and drops one of them (whichever 
                    comes first in the column sequence)
        It detects duplicate rows and drops one of them or keeps only one copy 
                    of duplicate rows
        It detects duplicate columns and drops one of them or keeps only one copy
        It detects skewed distributions and applies log or box-cox 
                    transformations on them
        It detects imbalanced classes and leaves them as it is
        It detects feature leakage and drops one of those features if 
                    they are highly correlated to target
    """
    def __init__(self, quantile=0.87, cat_fill_value = 'missing', 
                num_fill_value = 9999, rare_threshold = 0.01, 
                correlation_threshold = 0.9):
        super().__init__()  # Call the parent class constructor
        # Additional initialization code here
        self.quantile = quantile
        self.cat_fill_value = cat_fill_value
        self.num_fill_value = num_fill_value
        self.rare_threshold = rare_threshold
        self.correlation_threshold = correlation_threshold

###################################################################################
from pandas_dq import dq_report
def data_cleaning_suggestions(df, target=None):
    """
    This is a simple program to give data cleaning and improvement suggestions in class AV.
    Make sure you send in a dataframe. Otherwise, this will give an error.
    """
    if isinstance(df, pd.DataFrame):
        dqr = dq_report(data=df, target=target, html=False, csv_engine="pandas", verbose=1)
    else:
        print("Input must be a dataframe. Please check input and try again.")
    return dqr
###################################################################################
