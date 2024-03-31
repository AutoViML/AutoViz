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
    "pandas",
    "pyamg",
    "matplotlib<=3.7.4",  # Specify versions compatible with older Python versions
    "seaborn>=0.12.2",
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

if python_version < (3, 10):
    install_requires = list_req + [
    "numpy<1.25.0",  
    "hvplot~=0.7.3",      # Older compatible version
    "holoviews~=1.14.9",  # Older compatible version
    "panel~=0.14.4", ## this is an old versjon of panel
    "param==1.13.0", ### something broke in panel without this
    ]

# For Python versions >= 3.10 and < 3.11, update the dependency list
if (3, 10) <= python_version < (3, 11):
    install_requires = list_req + [
        # Keep most dependencies as is, adjust only where necessary
        "numpy>=1.25.0",  # Update as needed for compatibility with newer HoloViews
        # Update other dependencies as needed
        "hvplot>=0.9.2", ###newer hvplot
        "holoviews>=1.16.0",  # Update based on the bug fix relevant to Python 3.10
        # Ensure other dependencies are compatible
        "panel>=1.4.0",
    ]

# For Python versions >= 3.11, ensure HoloViews is at least 1.15.3 for the bug fix
if python_version >= (3, 11):
    install_requires = list_req + [
        # Adjust dependencies as needed for Python 3.11
        "numpy>=1.25.0",  # Update as needed for compatibility with newer HoloViews
        "hvplot>=0.9.2", ###newer hvplot
        "holoviews>=1.15.3",  # Ensure version is >= 1.15.3 for Python 3.11 support
        # Update or keep other dependencies as needed
        "panel>=1.4.0",
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