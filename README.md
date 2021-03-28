# Mobility in Europe during Covid-19 Pandemic
## University of Bonn WS 20/21 Effective Programming Practices for Economists Course Final Project by Bahar Coskun, Timo Haller and Anto Marcinkovic

We analyze the changes in peoples mobility during Covid-19 pandemic. We look at how people
change their movement with changing Covid-19 case numbers and different polices to this crisis.
We look at European countries and in particular Germany which we analyze also in state level.
We rely on mobility data made accessible by Google, stringency index and infection numbers which we collected from Our World in Data.

### Getting started
In order to run this project in your local machine you need to have a release of Python which is the programming language we relied on, a modern LaTex distribution in order to compile .tex documents, estimagic for graphics,
geckodriver for webscraping. The project is created and run on MacOS Big Sur version 11.2.

1. We recommend to download Anaconda Navigator for getting started with Python: https://www.anaconda.com
2. LaTex distribution such as TeXLive, MacTex, or MikTex
3. Geckodriver can be downloaded from https://github.com/mozilla/geckodriver/releases and geckodriver.exe should be added to the PATH
on a macbook this can simply be done by moving .exe in to usr/bin/local.
4. Install Estimagic by $ conda config --add channels conda-forge $ conda install -c opensourceeconomics estimagic : https://estimagic.readthedocs.io/en/latest/getting_started/installation.html
5. Before running the project create and activate the envrionment by: $ conda env create -f environment.yml and then $ conda activate covid_19_mobility
6. We rely on pytask to run the project once the project is cloned and all above steps are completed $ conda develop .
$ pytask

### Project Structure
src folder includes all the necessary code needed for the analysis:
1. data_management folder includes the code to scrape the data, clean and format it for analysis
2. analysis folder includes the code for plots and regressions
3. documentation folder is where you can find the information on our code
4. paper folder is where we present our findings, figures and tables
5. sandbox is where we include an interactive Jupyter notebook for data management and results
6. utils.py includes the the small function we use accross the project.
7. test_moving_avg.py tests whether we calculate forward moving average correctly. As we use forward moving averages of data for our analysis, this step is taken to 
ensure there are no calculation mistakes.
=======

