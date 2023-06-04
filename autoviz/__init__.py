name = "autoviz"
from .__version__ import __version__, __holo_version__
from .AutoViz_Class import AutoViz_Class
from .AutoViz_Class import data_cleaning_suggestions
from .AutoViz_Class import FixDQ
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""%s v%s. After importing autoviz, execute '%%matplotlib inline' to display charts inline.
    AV = AutoViz_Class()
    dfte = AV.AutoViz(filename, sep=',', depVar='', dfte=None, header=0, verbose=1, lowess=False,
               chart_format='svg',max_rows_analyzed=150000,max_cols_analyzed=30, save_plot_dir=None)""" %(module_type, version_number))
###########################################################################################
