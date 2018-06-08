# README #
## Overview ##

Bottle_air is a server which takes in a feature file (only CSVs are supported for now) and runs a hyperparameter optimization algorithm over deep network architectures to find the optimal candidate. Once the appropriate candidate is found, it is extensively trained, and saved for inference.

Once the model is saved, the user can access the model and run new data for predictions.

The server is written in Python 2.7 using Bottle (TODO: migrate to Flask or Django). The server is completely JSON based apart from static HTMLs. Every query to the server can be made as a POST request. No authentication has been added **yet**.

## Installing and Running ##

A bit of docs to get the server up-and-running:

Bottle_air runs on python 2.7. It uses the Bottle framework and the CherryPy server to support multithreading. The project is meant to be run under a virtualenv. The following steps should get it running:

(Assuming Ubuntu)

* Get git, virtualenv.
* git clone https://github.com/Apaezmx/simplifai.git
* cd simplifai
* virtualenv .
* source bin/activate
* pip install --upgrade pip
* pip install -r reqs.txt
* Install system libraries until previous command finishes successfully.
* If using tensorflow with GPU then pip uninstall tensoflow followed by pip install tensorflow-gpu (needs Cuda and cuDNN).
* python bottle_air/server.py

Once the server is running it should be available at http://localhost:8080/. If you need to change the server's port, just change it at the end of server.py. There are test CSVs to try under pusher/air/test_files

## Code guidelines ##

Try to comment as much as possible. **Follow existing standards**. Line length is set to 120 characters.

## Git guidelines ##

Main work is on master branch. Try to submit atomic, fully functioning changes. Test the server before every commit. No code review necessary yet. Remember to update reqs.txt if you add dependencies. Do not upload data files bigger than 10 MB.

**If you are merging from a local branch, try to squash commits so that you do not upload merge commits.**
