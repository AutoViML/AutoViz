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
    "numpy<1.25.0",  
    "pandas",
    "pyamg",
    "matplotlib<=3.7.4",  # Specify versions compatible with older Python versions
    "seaborn>=0.12.2",
    "scikit-learn",
    "statsmodels",
    "nltk",
    "textblob",
    "holoviews~=1.14.9",  # Older compatible version
    "bokeh~=2.4.2",       # Ensure compatibility with older HoloViews
    "hvplot~=0.7.3",      # Older compatible version
    "panel>=0.12.6",
    "xgboost>=0.82,<1.7",
    "fsspec>=0.8.3",
    "typing-extensions>=4.1.1",
    "pandas-dq>=1.29"
]
# Define default dependencies (compatible with older Python versions)
install_requires = list_req

# For Python versions >= 3.10 and < 3.11, update the dependency list
if (3, 10) <= python_version < (3, 11):
    install_requires = list_req + [
        # Keep most dependencies as is, adjust only where necessary
        "numpy>=1.25.0",  # Update as needed for compatibility with newer HoloViews
        # Update other dependencies as needed
        "holoviews>=1.16.0",  # Update based on the bug fix relevant to Python 3.10
        # Ensure other dependencies are compatible
    ]

# For Python versions >= 3.11, ensure HoloViews is at least 1.15.3 for the bug fix
if python_version >= (3, 11):
    install_requires = list_req + [
        # Adjust dependencies as needed for Python 3.11
        "holoviews>=1.15.3",  # Ensure version is >= 1.15.3 for Python 3.11 support
        # Update or keep other dependencies as needed
    ]

setuptools.setup(
    name="autoviz",
    version="0.1.807",
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