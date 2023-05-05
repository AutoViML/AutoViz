# AutoViz: The One-Line Automatic Data Visualization Library

![logo](logo.png)

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

# Motivation
<p>The motivation to create AutoViz stems from the need for a more efficient, user-friendly, and automated approach to data visualization. Visualizing data is a crucial step in the data analysis process, as it helps users understand patterns, trends, and relationships in the data. However, creating insightful visualizations can be time-consuming and require specialized knowledge of various plotting libraries and techniques.</p>

<p>AutoViz addresses these challenges by providing an easy-to-use, automated solution for generating meaningful visualizations with minimal effort. Its key motivation is to:</p>

<ol>
  <li><strong>Save time and effort</strong>: AutoViz simplifies the visualization process by requiring just a single line of code to generate multiple insightful plots, eliminating the need to write multiple lines of code for each chart.</li>
  <li><strong>Handle large datasets</strong>: AutoViz is designed to work with datasets of any size, intelligently sampling the data when necessary to ensure that the visualizations are generated quickly and efficiently, without compromising on the insights.</li>
  <li><strong>Accessibility</strong>: AutoViz makes data visualization accessible to a broader audience, including non-experts and beginners in data analysis, by abstracting away the complexities of various plotting libraries.</li>
  <li><strong>Automate the visualization process</strong>: AutoViz intelligently selects the appropriate visualizations for the given data, taking into account the data types and relationships among variables, which helps users quickly gain insights without having to manually decide which plots to create.</li>
  <li><strong>Customization and interactivity</strong>: AutoViz offers various options for customization, enabling users to tailor the generated visualizations to their specific needs and preferences. Moreover, with interactive chart formats like Bokeh, users can explore the data more dynamically.</li>
</ol>

<p>In summary, the motivation behind AutoViz is to make data visualization more efficient, accessible, and automated, enabling users to quickly gain valuable insights from their data and focus on making data-driven decisions.</p>

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
# Usage
Discover how to use AutoViz in this <a href="https://towardsdatascience.com/autoviz-a-new-tool-for-automated-visualization-ec9c1744a6ad?gi=822a01be6cd0">Medium article</a>.

In the AutoViz directory, open a Jupyter Notebook or on the command terminal and use this line to instantiate the AutoViz_Class. <b>Note: You no longer need to use from autoviz.AutoViz_Class import AutoViz_Class</b>. Instead, simply run:

```
from autoviz import AutoViz_Class
AV = AutoViz_Class()
dft = AV.AutoViz(filename)
```

Feed AutoViz any input filename (in CSV, txt, or JSON format), and set `max_rows_analyzed` and `max_cols_analyzed` based on memory limitations in your environment, and watch the magic happen!
<p>AutoViz generates charts in multiple formats using the `chart_format` setting which can one of the following:
- `'png'`, `'svg'`, or `'jpg'`: Inline Matplotlib charts
    * Save locally (`verbose=2`) or display in Jupyter Notebooks (`verbose=1`)
    * Default AutoViz behavior
- `'bokeh'`: Interactive Bokeh charts in Jupyter Notebooks
- `'server'`: Browser-based dashboards with individual chart types
- `'html'`: Interactive Bokeh charts saved as HTML files under the `AutoViz_Plots` directory (working folder) or specified directory using the `save_plot_dir` setting

AutoViz takes care of the rest. You'll see charts and plots on your screen.

![var_charts](var_charts.JPG)

AV.AutoViz is the main plotting function in AutoViz_Class (AV). Depending on the chart_format you choose, AutoViz will automatically call either the AutoViz_Main function or AutoViz_Holo function.

# API
Arguments for AV.AutoViz() method:

- `filename`: Use an empty string ("") if there's no associated filename and you want to use a dataframe; set dfte as the dataframe name. Provide a filename and leave dfte empty to load the dataset.
- `sep`: File separator (comma, semi-colon, tab, or any column-separating value).
- `depVar`: Target variable in your dataset; leave empty if not applicable.
- `dfte`: Input dataframe for plotting charts; leave empty if providing a filename.
- `header`: Row number of the header row in your file (0 for the first row).
- `verbose`: 0 for minimal info and charts, 1 for more info and charts, or 2 for saving charts locally without display.
- `lowess`: Use regression lines for each pair of continuous variables against the target variable in small datasets; avoid using for large datasets (>100,000 rows).
- `chart_format`: `'svg', 'png', 'jpg', 'bokeh', 'server', or 'html'` for displaying or saving charts in various formats, depending on the `verbose` option.
- `max_rows_analyzed`: Limit the max number of rows for chart display, particularly useful for very large datasets (millions of rows) to reduce chart generation time. A statistically valid sample will be used.
- `max_cols_analyzed`: Limit the number of continuous variables to be analyzed.
- `save_plot_dir`: Directory for saving plots. Default is None, which saves plots under the current directory in a subfolder named AutoViz_Plots. If the save_plot_dir doesn't exist, it will be created.

![bokeh_charts](bokeh_charts.JPG)

## Additional Notes
<ul>
  <li>AutoViz will visualize any sized file using a statistically valid sample.</li>
  <li>COMMA is the default separator in the file, but you can change it.</li>
  <li>Assumes the first row as the header in the file, but this can be changed.</li>
</ul>

<h3>Verbose Option</h3>
<ol>
  <li>Display minimal information but show charts in your notebook</li>
  <li>Print extra information on the notebook and display charts</li>
  <li>Do not display charts; save them locally under the AutoViz_Plots directory in your current working folder.</li>
</ol>

<h3>Chart Format Option</h3>
<ul>
  <li><code>'svg', 'jpg' or 'png'</code>: Display or save all charts, depending on the verbose option</li>
  <li><code>'bokeh'</code>: Plot interactive charts using Bokeh in your Jupyter Notebook</li>
  <li><code>'server'</code>: Display charts in your browser, with one chart type per tab</li>
  <li><code>'html'</code>: Create interactive Bokeh charts and save them under the AutoViz_Plots directory or the directory specified in the save_plot_dir setting.</li>
</ul>

![server_charts](server_charts.JPG)

# Examples
Here are some examples to help you get started with AutoViz:

## Example 1: Visualize a CSV file with a target variable

```
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

## Example 2: Visualize a Pandas DataFrame without a target variable:

```
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
    chart_format="svg",
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir=None
)
```

## Example 3: Generate interactive Bokeh charts and save them as HTML files in a custom directory

```
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
    chart_format="html",
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir=custom_plot_dir
)
```

These examples should give you an idea of how to use AutoViz with different scenarios and settings. By tailoring the options and settings, you can generate visualizations that best suit your needs, whether you're working with large datasets, interactive charts, or simply exploring the relationships between variables.

# Maintainers
AutoViz is actively maintained and improved by a team of dedicated developers. If you have any questions, suggestions or issues, feel free to reach out to the maintainers.
* [@AutoViML](https://github.com/AutoViML)
* [@morenoh149](https://github.com/morenoh149)
* [@hironroy](https://github.com/hironroy)


# Contributing
We welcome contributions from the community! If you're interested in contributing to AutoViz, please follow these steps:

<h2>Fork the repository on GitHub</h2>
<ol>
  <li>Clone your fork and create a new branch for your feature or bugfix.</li>
  <li>Commit your changes to the new branch, ensuring that you follow coding standards and write appropriate tests.</li>
  <li>Push your changes to your fork on GitHub.</li>
  <li>Submit a pull request to the main repository, detailing your changes and referencing any related issues.</li>
</ol>

See [the contributing file](contributing.md)!

# License
AutoViz is released under the Apache License, Version 2.0. By using AutoViz, you agree to the terms and conditions specified in the license.

# Tips
<p>That covers the main aspects of AutoViz, but here are some additional tips and reminders to help you make the most of the library:</p>
<ul>
  <li>Make sure to regularly upgrade AutoViz to benefit from the <a href="https://github.com/AutoViML/AutoViz/blob/main/updates.md">latest features, bug fixes, and improvements</a>. You can update it using <code>pip install --upgrade autoviz</code>.</li>
  <li>AutoViz is highly customizable, so don't hesitate to explore and experiment with various settings, such as <code>chart_format</code>, <code>verbose</code>, and <code>max_rows_analyzed</code>. This will allow you to create visualizations that better suit your specific needs and preferences.</li>
  <li>Remember to delete the <code>AutoViz_Plots</code> directory (or any custom directory you specified) periodically if you used the <code>verbose=2</code> option, as it can accumulate a large number of saved charts over time.</li>
  <li>For further guidance or inspiration, check out the <a href="https://towardsdatascience.com/autoviz-a-new-tool-for-automated-visualization-ec9c1744a6ad">Medium article on AutoViz</a>, as well as other online resources and tutorials.</li>
  <li>If you encounter any issues, have questions, or want to suggest improvements, don't hesitate to engage with the AutoViz community through the <a href="https://github.com/AutoViML/AutoViz">GitHub repository</a> or other online platforms.</li>
</ul>
<p>By leveraging AutoViz's powerful and flexible features, you can streamline your data visualization process and gain valuable insights more efficiently. Happy visualizing!</p>

## DISCLAIMER
This project is not an official Google project. It is not supported by Google and Google specifically disclaims all warranties as to its quality, merchantability, or fitness for a particular purpose.
