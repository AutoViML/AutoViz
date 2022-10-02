name = "autoviz"
from .__version__ import __version__, __holo_version__
from .AutoViz_Class import AutoViz_Class
from .classify_method import data_cleaning_suggestions
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""%s v%s. After importing, execute '%%matplotlib inline' to display charts in Jupyter.
    AV = AutoViz_Class()
    dfte = AV.AutoViz(filename, sep=',', depVar='', dfte=None, header=0, verbose=1, lowess=False,
               chart_format='svg',max_rows_analyzed=150000,max_cols_analyzed=30, save_plot_dir=None)""" %(module_type, version_number))
print("Update: verbose=0 displays charts in your local Jupyter notebook.")
print("        verbose=1 additionally provides EDA data cleaning suggestions. It also displays charts.")
print("        verbose=2 does not display charts but saves them in AutoViz_Plots folder in local machine.")
print("        chart_format='bokeh' displays charts in your local Jupyter notebook.")
print("        chart_format='server' displays charts in your browser: one tab for each chart type")
print("        chart_format='html' silently saves interactive HTML files in your local machine")
###########################################################################################
