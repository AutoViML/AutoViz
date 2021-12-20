import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoviz",
    version="0.1.29",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Automatically Visualize any dataset, any size with a single line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/AutoViz",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "ipython",
        "jupyter",
        "xlrd",
        "wordcloud",
        "pandas",
        "matplotlib>=3.3.3",
        "seaborn>=0.11.1",
        "scikit-learn",
        "statsmodels",
        "holoviews==1.14.6",
        "bokeh==2.4.2",
        "hvplot==0.7.3",
        "panel==0.12.6",
        "xgboost",
        "fsspec==0.8.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
