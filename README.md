This workflow is mainly designed to be run using SLURM on the AC cluster, but changes can be made to run elsewhere.

!!! VERY IMPORTANT !!!
Set up environment variables (from window directory):

$ export wpath=$PWD
$ export util=utils/utils1.json
$ export dpath=/path/to/MAFAULDA/data/
$ export vpath=utils/test_files.csv
$ export model=models/rfc1.joblib


How to set up python virtual environment:

$ python3 -m venv window-env
$ source window-env/bin/activate
$ python3 -m pip install --upgrade pip
$ python3 -m pip install pandas scipy numpy matplotlib seaborn scikit-learn pyyaml


Making a config file:

The base template for a config file can be found in window/config/config.yaml. config["default"]["experiment_path"] determines where plots and metadata will be saved.

Usage:

$ export config=config/{config_file_of_your_choice}

$ python run.py $wpath $config


Usage for several config files:

Modify script.sh to include the config files you want and then run
$ ./script.sh


After completing the above steps, you can try testing the workflow by running the following:

$ python run.py $wpath $config

There should be 250 jobs running in the queue if everything went right


What gets generated by each run?

An out file, named "out", can be found in plots/{experiment_name}/{windowWidth_widthPerStep} and contains accuracy and timing information. To condense all out files into one, use collect.py

collect.py usage:

python collect.py plots/{experiment1} plots/{experiment2} plots/{experimentN}

Any number of experiment directories can be provided as command line arguments to this script. All this script does is combine each out file from any given directory and place a new out file in the base directory.
