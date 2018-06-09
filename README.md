# README #
<img src="http://www.simplifai.mx/wp-content/uploads/2017/05/SimplifAiLogo_1.png" alt="SimpifAi-Simplifying Technology"/>
<iframe width="560" height="315" src="https://www.youtube.com/embed/o0ix881crP8" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
## Demo ##
http://www.simplifai.mx:8012/
### How to Use ###
Open the demo link and click on the start button. 
Download the human resources example CSV and take a look at it. You should see the format which consists of the first row being the headers (name of the columns) and the rest of the rows are just the data. One final detail to notice is that the column we want the model to predict is prefixed by "output_" (TODO add a more friendly way of doing this).
So going back to the browser, upload the HR file and click continue. You'll see all the columns parsed and ready to be trained on.
Click on continue and the training will start. After a few minutes training will be done and you'll be redirected to the infer site (Enable pop-ups on your browser). There you can now use the trained model.


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

You can also just run the start.sh file.

Once the server is running it should be available at http://localhost:8080/. If you need to change the server's port, just change it at the end of server.py. There are test CSVs to try under simplifai/bottle_air/test_files

## Code guidelines ##

Try to comment as much as possible. **Follow existing standards**. Line length is set to 120 characters.

## Git guidelines ##

Main work is on master branch. Try to submit atomic, fully functioning changes. Test the server before every commit. No code review necessary yet. Remember to update reqs.txt if you add dependencies. Do not upload data files bigger than 10 MB.

**If you are merging from a local branch, try to squash commits so that you do not upload merge commits.**
