# AutoViz
Automatically Visualize any dataset, any size with a single line of code

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
    #######             AutoViz Class by Ram Seshadri                       ######
    #######      AUTOMATICALLY VISUALIZE ANY DATA SET                       ######
    #######            V3.0 6/15/19 Version                                 ######
    ##############################################################################
    ##### AUTOVIZ PERFORMS AUTOMATIC VISUALIZATION OF ANY DATA SET WITH ONE CLICK.
    #####    Give it any input file (CSV, txt or json) and it visualize it for u.
    ##### INPUTS:                                                            #####
    ##### A FILE NAME OR A DATA FRAME AS INPUT.                              #####
    ##### AutoViz will visualize any sized file using a statistically valid sample.
    #####  - COMMA is assumed as default separator in file. But u can change it.##
    #####  - Assumes first row as header in file but you can change it.      #####
    #####  - First instantiate an AutoViz class to  hold output of charts, plots.#
    #####  - Then call the Autoviz program with inputs as defined below.       ###
    ##############################################################################
 <h1><a class="h" name="DOWNLOAD-INSTALLATION" href="#DOWNLOAD-INSTALLATION"><span></span></a><a class="h" name="download-installation" href="#download-installation"><span></span></a>DOWNLOAD / INSTALLATION</h1><ol><li>Copy or download this entire directory of files to any local directory using git clone or any download methods.</li></ol><h1><a class="h" name="RUN-AUTOViZ" href="#RUN-AUTOViZ"><span></span></a><a class="h" name="run-autoviz" href="#run-autoviz"><span></span></a>RUN AutoViz</h1><ol start="2"><li><p>In the same directory, open a Jupyter Notebook and use this line to import the .py file: <br>from AutoViz_Class import *<br> AV = AutoViz() </p></li><li><p>Load a data set (any CSV or text file) into a Pandas dataframe or give the name of the path and filename you want to visualize. If you don't have a filename, you can simply assign the filename variable below to '' (empty string):</p></li></ol><li><p>Finally, call AutoViz using the filename (or dataframe) along with the separator (if any in file) and the name of the target variable in file or data frame. That's all. AutoViz will do the rest. You will see charts and plots on your screen.</p></li></ol><p><br>df = AV.autoviz('','',target,df,verbose=0)</p><h1><a class="h" name="DISCLAIMER" href="#DISCLAIMER"><span></span></a><a class="h" name="disclaimer" href="#disclaimer"><span></span></a>DISCLAIMER</h1><p>“This is not an official Google product”.</p><h1><a class="h" name="LICENSE" href="#LICENSE"><span></span></a><a class="h" name="license" href="#license"><span></span></a>LICENSE</h1><p>Licensed under the Apache License, Version 2.0 (the &ldquo;License&rdquo;).</p></div></div></div><!-- default customFooter --><footer class="Site-footer"><div class="Footer"><span class="Footer-poweredBy">Powered by <a href="https://gerrit.googlesource.com/gitiles/">Gitiles</a>| <a href="https://policies.google.com/privacy">Privacy</a></span><div class="Footer-links"></div></div></footer></body></html>
