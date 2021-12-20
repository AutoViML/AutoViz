name = "autoviz"
from .__version__ import __version__, __holo_version__
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Alert! from'
version_number = __version__
print("""%s autoviz version %s, after importing, you must %%matplotlib inline to display charts in Jupyter Notebooks.
    AV = AutoViz_Class()
    AV.AutoViz(filename, sep=',', depVar='', dfte=None, header=0, verbose=0, lowess=False,
               chart_format='svg',max_rows_analyzed=150000,max_cols_analyzed=30, save_plot_dir=None)""" %(module_type, version_number))
print("Note: verbose=0 or 1 generates charts and displays them in your local Jupyter notebook.")
print("      verbose=2 does not display plots but saves them in AutoViz_Plots folder in local machine.")
print("Note: chart_format='bokeh' generates and displays charts in your local Jupyter notebook.")
print("      chart_format='server' generates and displays charts in the browser - one tab for each chart.")
print("      chart_format='html' silently saves charts HTML format - they are also interactive!")
###########################################################################################

