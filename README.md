# README #

A bit of docs to get the server up-and-running:

Air runs on python 2.7. It uses the Bottle framework and the CherryPy server to support multithreading. The project is meant to be run under a virtualenv. The following steps should get it running:
(Assuming Ubuntu)
* Get git, virtualenv.
* git clone https://<me>@bitbucket.org/Apaezmx/pusher.git
* cd pusher
* virtualenv .
* source bin/activate
* pip install --upgrade pip
* pip install -r reqs.txt
* Install system libraries until previous command finishes successfully.
* If using tensorflow with GPU then pip uninstall tensoflow followed by pip install tensorflow-gpu (needs Cuda and cuDNN).
* python air/server.py

Once the server is running it should be available at http://localhost:8080/. If you need to change the server's port, just change it at the end of server.py.