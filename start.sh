sudo apt-get --assume-yes install python-pip python-dev build-essential git libev-dev libffi-dev libssl-dev libxml2-dev libxslt1-dev zlib1g-dev

sudo pip install virtualenv virtualenvwrapper

git clone https://Apaezmx@bitbucket.org/Apaezmx/pusher.git

cd pusher

mkdir air/models

virtualenv --distribute .

source bin/activate

pip install --upgrade pip

pip install -r reqs.txt

easy_install bottle_memcache
