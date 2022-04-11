This workflow is mainly designed to be run on the AC cluster, but changes can be made to run elsewhere.

Set up environment variables (from window directory):

$ export wpath=$PWD
$ export util=utils/utils1.json
$ export dpath=/path/to/MAFAULDA/data/


How to set up python virtual environment:

$ python3 -m venv window-env
$ source window-env/bin/activate
$ python3 -m pip install --upgrade pip
$ python3 -m pip install pandas scipy numpy matplotlib seaborn scikit-learn yaml


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
