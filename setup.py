import setuptools
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

# Determine the Python version
python_version = sys.version_info

list_req = [
    "xlrd",
    "wordcloud",
    "emoji",
    # Assuming numpy version <1.25.0 is compatible with older Python versions and older HoloViews
    "pyamg",
    "scikit-learn",
    "statsmodels",
    "nltk",
    "textblob",
    "xgboost>=0.82,<1.7",
    "fsspec>=0.8.3",
    "typing-extensions>=4.1.1",
    "pandas-dq>=1.29"
]
# Define default dependencies (compatible with older Python versions)
install_requires = list_req

# Define default dependencies (compatible with older Python versions)
install_requires = list_req + [
    # Keep most dependencies as is, adjust only where necessary
    "numpy>=1.24.0",  # Update as needed for compatibility with newer HoloViews
    # Update other dependencies as needed
    "hvplot>=0.9.2", ###newer hvplot
    "holoviews>=1.16.0",  # Update based on the bug fix relevant to Python 3.10
    # Ensure other dependencies are compatible
    "panel>=1.4.0", ## this is a new version of panel
    "pandas>=2.0", ## pandas must be below 2.0 version
    "matplotlib>3.7.4", ## newer version of matplotlib
    "seaborn>0.12.2", ## newer version of seaborn ##
]

setuptools.setup(
    name="autoviz",
    version="0.1.903",
    author="Ram Seshadri",
    description="Automatically Visualize any dataset, any size with a single line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/AutoViz.git",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)