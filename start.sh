sudo apt-get --assume-yes install python-pip python-dev build-essential git libev-dev libffi-dev libssl-dev libxml2-dev libxslt1-dev zlib1g-dev

sudo pip install virtualenv virtualenvwrapper

cd ~

git clone https://github.com/Apaezmx/simplifai.git

cd simplifai

mkdir bottle_air/models

virtualenv --distribute .

source bin/activate

pip install --upgrade pip

pip install -r reqs.txt

easy_install bottle_memcache
