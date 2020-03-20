# AutoViz

![banner](logo.png)

[![Pepy Downloads](https://pepy.tech/badge/autoviz)](https://pepy.tech/project/autoviz)
[![Pepy Downloads per week](https://pepy.tech/badge/autoviz/week)](https://pepy.tech/project/autoviz)
[![Pepy Downloads per month](https://pepy.tech/badge/autoviz/month)](https://pepy.tech/project/autoviz)
[![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg)](https://github.com/RichardLitt/standard-readme)
[![Python Versions](https://img.shields.io/pypi/pyversions/autoviz.svg)](https://pypi.org/project/autoviz)
[![PyPI Version](https://img.shields.io/pypi/v/autoviz.svg)](https://pypi.org/project/autoviz)
[![PyPI License](https://img.shields.io/pypi/l/autoviz.svg)](https://github.com/AutoViML/AutoViz/blob/master/LICENSE)

Automatically Visualize any dataset, any size with a single line of code.

AutoViz performs automatic visualization of any dataset with one line.
Give any input file (CSV, txt or json) and AutoViz will visualize it.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Install

**Prerequsites**

- [Anaconda](https://docs.anaconda.com/anaconda/install/)

To clone AutoViz, it's better to create a new environment, and install the required dependencies:

To install from PyPi:

```sh
conda create -n <your_env_name> python=3.7 anaconda
conda activate <your_env_name> # ON WINDOWS: `source activate <your_env_name>`
pip install autoviz
```

To install from source:

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

Read this Medium article to know how to use [AutoViz](https://towardsdatascience.com/autoviz-a-new-tool-for-automated-visualization-ec9c1744a6ad).

In the AutoViz directory, open a Jupyter Notebook and use this line to instantiate the library

```py
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
```

Load a dataset (any CSV or text file) into a Pandas dataframe or give the name of the path and filename you want to visualize.
If you don't have a filename, you can simply assign the filename argument `""` (empty string).

Call AutoViz using the filename (or dataframe) along with the separator and the name of the target variable in the input.
AutoViz will do the rest. You will see charts and plots on your screen.

```py
filename = ""
sep = ","
dft = AV.AutoViz(
    filename,
    sep,
    target,
    df,
    header=0,
    verbose=0,
    lowess=False,
    chart_format="svg",
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
)
```

This is the main calling program in AV.
It will call all the load, display and save programs that are currently outside AV.
This program will draw scatter and other plots for the input dataset and then call the correct variable name with the `add_plots` function and send in the chart created by that plotting program, for example, scatter.
You have to make sure that `add_plots` function has the exact name of the variable defined in the Class AV.
If not, this will give an error.

**Notes:**

* AutoViz will visualize any sized file using a statistically valid sample.
* `COMMA` is assumed as default separator in file. But you can change it.
* Assumes first row as header in file but you can change it.

## API

**Arguments**

- `max_rows_analyzed` - limits the max number of rows that is used to display charts
- `max_cols_analyzed` - limits the number of continuous vars that can be analyzed
- `verbose`
  - if 0, does not print any messages and goes into silent mode. This is the default.
  - if 1, print messages on the terminal and also display charts on terminal.
  - if 2, print messages but will not display charts, it will simply save them.

## Maintainers

* [@AutoViML](https://github.com/AutoViML)
* [@morenoh149](https://github.com/morenoh149)
* [@hironroy](https://github.com/hironroy)

## Contributing

See [the contributing file](contributing.md)!

PRs accepted.

## License

Apache License, Version 2.0 

## DISCLAIMER
This project is not an official Google project. It is not supported by Google and Google specifically disclaims all warranties as to its quality, merchantability, or fitness for a particular purpose.

