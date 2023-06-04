# AutoViz: The One-Line Automatic Data Visualization Library

![logo](images/logo.png)

Unlock the power of **AutoViz** to visualize any dataset, any size, with just a single line of code! Plus, now you can save these interactive charts as HTML files automatically with the `html` setting.

[![Pepy Downloads](https://pepy.tech/badge/autoviz)](https://pepy.tech/project/autoviz)
[![Pepy Downloads per week](https://pepy.tech/badge/autoviz/week)](https://pepy.tech/project/autoviz)
[![Pepy Downloads per month](https://pepy.tech/badge/autoviz/month)](https://pepy.tech/project/autoviz)
[![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg)](https://github.com/RichardLitt/standard-readme)
[![Python Versions](https://img.shields.io/pypi/pyversions/autoviz.svg)](https://pypi.org/project/autoviz)
[![PyPI Version](https://img.shields.io/pypi/v/autoviz.svg)](https://pypi.org/project/autoviz)
[![PyPI License](https://img.shields.io/pypi/l/autoviz.svg)](https://github.com/AutoViML/AutoViz/blob/master/LICENSE)

With AutoViz, you can easily and quickly generate insightful visualizations for your data. Whether you're a beginner or an expert in data analysis, AutoViz can help you explore your data and uncover valuable insights. Try it out and see the power of automated visualization for yourself!

## Table of Contents

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

## Motivation
The motivation to create AutoViz stems from the need for a more efficient, user-friendly, and automated approach to data visualization. Visualizing data is a crucial step in the data analysis process, as it helps users understand patterns, trends, and relationships in the data. However, creating insightful visualizations can be time-consuming and require specialized knowledge of various plotting libraries and techniques.

AutoViz addresses these challenges by providing an easy-to-use, automated solution for generating meaningful visualizations with minimal effort. Its key motivations are:

1. **Save time and effort**: AutoViz simplifies the visualization process by requiring just a single line of code to generate multiple insightful plots, eliminating the need to write multiple lines of code for each chart.

2. **Handle large datasets**: AutoViz is designed to work with datasets of any size, intelligently sampling the data when necessary to ensure that the visualizations are generated quickly and efficiently, without compromising on the insights.

3. **Democratize Data Science**: AutoViz makes data visualization accessible to a broader audience, including non-experts and beginners in data analysis, by abstracting away the complexities of various plotting libraries.

4. **Automate EDA**: AutoViz now automatically analyzes and fixes Data Quality issues in your dataset. This will help users to quickly go from insights to action without having to manually analyze each variable. AutoViz uses the new `pandas-dq` library created by `autoviml` to perform data quality and cleaning.

5. **Customization and interactivity**: AutoViz offers various options for customization, enabling users to tailor the generated visualizations to their specific needs and preferences. Moreover, with interactive chart formats like Bokeh, users can explore the data more dynamically.

In summary, the motivation behind AutoViz is to make data visualization more efficient, accessible, and automated, enabling users to quickly gain valuable insights from their data and focus on making data-driven decisions.

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

In the AutoViz directory, open a Jupyter Notebook or the command terminal and use the following code to instantiate the AutoViz_Class. Note: You no longer need to use ```from autoviz.AutoViz_Class import AutoViz_Class```. Instead, simply run:

```python
from autoviz import AutoViz_Class
AV = AutoViz_Class()
dft = AV.AutoViz(filename)
```

Feed AutoViz any input filename (in CSV, txt, or JSON format), and set `max_rows_analyzed` and `max_cols_analyzed` based on memory limitations in your environment, and watch the magic happen!

AutoViz can now create charts in multiple  formats using the `chart_format` setting:
- If `chart_format ='png'` or `'svg'` or `'jpg'`: Matplotlib charts are plotted inline.
    * Can be saved locally (using `verbose=2` setting) or displayed (`verbose=1`) in Jupyter Notebooks.
    * This is the default behavior for AutoViz.
- If `chart_format='bokeh'`: Interactive Bokeh charts are plotted in Jupyter Notebooks.
- If `chart_format='server'`, dashboards will pop up for each kind of chart on your browser.
- If `chart_format='html'`, interactive Bokeh charts will be created and silently saved as HTML files under the `AutoViz_Plots` directory (under working folder) or any other directory that you specify using the `save_plot_dir` setting (during input).


## API
Arguments for `AV.AutoViz()` method:

- `filename`: Use an empty string ("") if there's no associated filename and you want to use a dataframe; set dfte as the dataframe name. Provide a filename and leave dfte empty to load the dataset.
- `sep`: File separator (comma, semi-colon, tab, or any column-separating value).
- `depVar`: Target variable in your dataset; leave empty if not applicable.
- `dfte`: Input dataframe for plotting charts; leave empty if providing a filename.
- `header`: Row number of the header row in your file (0 for the first row).
- `verbose`: 0 for minimal info and charts, 1 for more info and charts, or 2 for saving charts locally without display.
- `lowess`: Use regression lines for each pair of continuous variables against the target variable in small datasets; avoid using for large datasets (>100,000 rows).
- `chart_format`: 'svg', 'png', 'jpg', 'bokeh', 'server', or 'html' for displaying or saving charts in various formats, depending on the verbose option.
- `max_rows_analyzed`: Limit the max number of rows for chart display, particularly useful for very large datasets (millions of rows) to reduce chart generation time. A statistically valid sample will be used.
- `max_cols_analyzed`: Limit the number of continuous variables to be analyzed.
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