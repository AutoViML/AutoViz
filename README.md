# AutoViz: The One-Line Automatic Data Visualization Library

![logo](images/logo.png)

Unlock the power of **AutoViz** to visualize any dataset, any size, with just a single line of code! Plus, now you can get a quick assessment of your dataset's quality and fix DQ issues through the FixDQ() function.

[![Pepy Downloads](https://pepy.tech/badge/autoviz)](https://pepy.tech/project/autoviz)
[![Pepy Downloads per week](https://pepy.tech/badge/autoviz/week)](https://pepy.tech/project/autoviz)
[![Pepy Downloads per month](https://pepy.tech/badge/autoviz/month)](https://pepy.tech/project/autoviz)
[![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg)](https://github.com/RichardLitt/standard-readme)
[![Python Versions](https://img.shields.io/pypi/pyversions/autoviz.svg)](https://pypi.org/project/autoviz)
[![PyPI Version](https://img.shields.io/pypi/v/autoviz.svg)](https://pypi.org/project/autoviz)
[![PyPI License](https://img.shields.io/pypi/l/autoviz.svg)](https://github.com/AutoViML/AutoViz/blob/master/LICENSE)

With AutoViz, you can easily and quickly generate insightful visualizations for your data. Whether you're a beginner or an expert in data analysis, AutoViz can help you explore your data and uncover valuable insights. Try it out and see the power of automated visualization for yourself!

## Table of Contents

- [Latest Updates for AutoViz](#Latest)
- [The purpose and motivation for AutoViz](#motivation)
- [How to install and setup AutoViz in your environment](#installation)
- [How to use AutoViz with various options and settings](#usage)
- [The API and available options](#api)
- [Examples to get you started](#examples)
- [Maintainers](#maintainers)
- [Contributing to the project](#contributing)
- [License](#license)
- [Additional Tips before you start](#tips)
- [Disclaimer](#disclaimer)

## Latest
The latest updates about `autoviz` library can be found in <a href="https://github.com/AutoViML/AutoViz/blob/master/updates.md">Updates page</a>.

## Motivation
The motivation behind the creation of AutoViz is to provide a more efficient, user-friendly, and automated approach to exploratory data analysis (EDA) through quick and easy data visualization plus data quality. The library is designed to help users understand patterns, trends, and relationships in the data by creating insightful visualizations with minimal effort. AutoViz is particularly useful for beginners in data analysis as it abstracts away the complexities of various plotting libraries and techniques. For experts, it provides another expert tool that they can use to provide inights into data that they may have missed.

AutoViz is a powerful tool for generating insightful visualizations with minimal effort. Here are some of its key selling points compared to other automated EDA tools:
<ol>
<li><b>Ease of use</b>: AutoViz is designed to be user-friendly and accessible to beginners in data analysis, abstracting away the complexities of various plotting libraries</li>
<li><b>Speed</b>: AutoViz is optimized for speed and can generate multiple insightful plots with just a single line of code</li>
<li><b>Scalability</b>: AutoViz is designed to work with datasets of any size and can handle large datasets efficiently</li>
<li><b>Automation</b>: AutoViz automates the visualization process, requiring just a single line of code to generate multiple insightful plots</li>
<li><b>Customization</b>: AutoViz provides several options for customizing the visualizations, such as changing the chart type, color palette, etc.</li>
<li><b>Data Quality</b>: AutoViz now provides data quality assessment by default and helps you fix DQ issues with a single line of code using the FixDQ() function</li>
</ol>
## Installation

**Prerequisites**
- [Anaconda](https://docs.anaconda.com/anaconda/install/)

Create a new environment and install the required dependencies to clone AutoViz:

**From PyPi:**
```sh
cd <AutoViz_Destination>
git clone git@github.com:AutoViML/AutoViz.git
# or download and unzip https://github.com/AutoViML/AutoViz/archive/master.zip
conda create -n <your_env_name> python=3.7 anaconda
conda activate <your_env_name> # ON WINDOWS: `source activate <your_env_name>`
cd AutoViz
pip install -r requirements.txt
```

## Usage
Discover how to use AutoViz in this Medium article.

In the AutoViz directory, open a Jupyter Notebook or open a command palette (terminal) and use the following code to instantiate the AutoViz_Class. You can simply run this code step by step:

```python
from autoviz import AutoViz_Class
AV = AutoViz_Class()
dft = AV.AutoViz(filename)
```

AutoViz can use any input either filename (in CSV, txt, or JSON format) or a pandas dataframe. If you have a large dataset, you can set the `max_rows_analyzed` and `max_cols_analyzed` arguments to speed up the visualization by asking autoviz to sample your dataset.

AutoViz can also create charts in multiple  formats using the `chart_format` setting:
- If `chart_format ='png'` or `'svg'` or `'jpg'`: Matplotlib charts are plotted inline.
    * Can be saved locally (using `verbose=2` setting) or displayed (`verbose=1`) in Jupyter Notebooks.
    * This is the default behavior for AutoViz.
- If `chart_format='bokeh'`: Interactive Bokeh charts are plotted in Jupyter Notebooks.
- If `chart_format='server'`, dashboards will pop up for each kind of chart on your browser.
- If `chart_format='html'`, interactive Bokeh charts will be created and silently saved as HTML files under the `AutoViz_Plots` directory (under working folder) or any other directory that you specify using the `save_plot_dir` setting (during input).


## API
Arguments for `AV.AutoViz()` method:

- `filename`: Use an empty string ("") if there's no associated filename and you want to use a dataframe. In that case, using the `dfte` argument for the dataframe. Otherwise provide a filename and leave `dfte` argument with an empty string. Only one of them can be used.
- `sep`: File separator (comma, semi-colon, tab, or any column-separating value) if you use a filename above.
- `depVar`: Target variable in your dataset; set it as an empty string if not applicable.
- `dfte`: name of the pandas dataframe for plotting charts; leave it as empty string if using a filename.
- `header`: set the row number of the header row in your file (0 for the first row). Otherwise leave it as 0.
- `verbose`: 0 for minimal info and charts, 1 for more info and charts, or 2 for saving charts locally without display.
- `lowess`: Use regression lines for each pair of continuous variables against the target variable in small datasets; avoid using for large datasets (>100,000 rows).
- `chart_format`: 'svg', 'png', 'jpg', 'bokeh', 'server', or 'html' for displaying or saving charts in various formats, depending on the verbose option.
- `max_rows_analyzed`: Limit the max number of rows to use for visualization when dealing with very large datasets (millions of rows). A statistically valid sample will be used by autoviz. Default is 150000 rows.
- `max_cols_analyzed`: Limit the number of continuous variables to be analyzed. Defaul is 30 columns.
- `save_plot_dir`: Directory for saving plots. Default is None, which saves plots under the current directory in a subfolder named AutoViz_Plots. If the save_plot_dir doesn't exist, it will be created.

## Examples
Here are some examples to help you get started with AutoViz. If you need full jupyter notebooks with code samples they can be found in [examples](https://github.com/AutoViML/AutoViz/tree/master/Examples) folder.

### Example 1: Visualize a CSV file with a target variable

```python
from autoviz import AutoViz_Class
AV = AutoViz_Class()

filename = "your_file.csv"
target_variable = "your_target_variable"

dft = AV.AutoViz(
    filename,
    sep=",",
    depVar=target_variable,
    dfte=None,
    header=0,
    verbose=1,
    lowess=False,
    chart_format="svg",
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir=None
)
```

![var_charts](images/var_charts.JPG)

### Example 2: Visualize a Pandas DataFrame without a target variable:

```python
import pandas as pd
from autoviz import AutoViz_Class

AV = AutoViz_Class()

data = {'col1': [1, 2, 3, 4, 5], 'col2': [5, 4, 3, 2, 1]}
df = pd.DataFrame(data)

dft = AV.AutoViz(
    "",
    sep=",",
    depVar="",
    dfte=df,
    header=0,
    verbose=1,
    lowess=False,
    chart_format="server",
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir=None
)

```

![server_charts](images/server_charts.JPG)

### Example 3: Generate interactive Bokeh charts and save them as HTML files in a custom directory

```python
from autoviz import AutoViz_Class
AV = AutoViz_Class()

filename = "your_file.csv"
target_variable = "your_target_variable"
custom_plot_dir = "your_custom_plot_directory"

dft = AV.AutoViz(
    filename,
    sep=",",
    depVar=target_variable,
    dfte=None,
    header=0,
    verbose=2,
    lowess=False,
    chart_format="bokeh",
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir=custom_plot_dir
)
```

![bokeh_charts](images/bokeh_charts.JPG)

These examples should give you an idea of how to use AutoViz with different scenarios and settings. By tailoring the options and settings, you can generate visualizations that best suit your needs, whether you're working with large datasets, interactive charts, or simply exploring the relationships between variables.

## Maintainers
AutoViz is actively maintained and improved by a team of dedicated developers. If you have any questions, suggestions, or issues, feel free to reach out to the maintainers:

- [@AutoViML](https://github.com/AutoViML)
- [@morenoh149](https://github.com/morenoh149)
- [@hironroy](https://github.com/hironroy)

## Contributing
We welcome contributions from the community! If you're interested in contributing to AutoViz, please follow these steps:

- Fork the repository on GitHub.
- Clone your fork and create a new branch for your feature or bugfix.
- Commit your changes to the new branch, ensuring that you follow coding standards and write appropriate tests.
- Push your changes to your fork on GitHub.
- Submit a pull request to the main repository, detailing your changes and referencing any related issues.

See [the contributing file](contributing.md)!

## License
AutoViz is released under the Apache License, Version 2.0. By using AutoViz, you agree to the terms and conditions specified in the license.

## Tips
Here are some additional tips and reminders to help you make the most of the library:

- **Make sure to regularly upgrade AutoViz** to benefit from the latest features, bug fixes, and improvements. You can update it using pip install --upgrade autoviz.
- **AutoViz is highly customizable, so don't hesitate to explore and experiment with various settings**, such as chart_format, verbose, and max_rows_analyzed. This will allow you to create visualizations that better suit your specific needs and preferences.
- **Remember to delete the AutoViz_Plots directory (or any custom directory you specified) periodically** if you used the verbose=2 option, as it can accumulate a large number of saved charts over time.
- **For further guidance or inspiration, check out the <a href="https://towardsdatascience.com/autoviz-a-new-tool-for-automated-visualization-ec9c1744a6ad">Medium article</a> on AutoViz**, as well as other online resources and tutorials.
<ul>
  <li>AutoViz will visualize any sized file using a statistically valid sample.</li>
  <li>COMMA is the default separator in the file, but you can change it.</li>
  <li>Assumes the first row as the header in the file, but this can be changed.</li>
</ul>

- **By leveraging AutoViz's powerful and flexible features**, you can streamline your data visualization process and gain valuable insights more efficiently. Happy visualizing!

## DISCLAIMER
This project is not an official Google project. It is not supported by Google, and Google specifically disclaims all warranties as to its quality, merchantability, or fitness for a particular purpose.