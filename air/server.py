import argparse
from api import resolve_endpoint
from bottle import error, post, request, route, run, static_file, template
from config import config
import os

@error(404)
def error404(error):
  print error
  return 'Nothing here, sorry'

@post('/api')
def serve_api():
  endpoint = request.forms.get('endpoint')
  return resolve_endpoint(endpoint, 
          {k: v for k, v in request.forms.iteritems()},
          {k: v for k, v in request.files.iteritems()})

@route('/')
def serve_index():
  return static_file('index.html', root=config.STATIC_PATH)

@route('/<filepath:path>')
def server_static(filepath):
  return static_file(filepath, root=config.STATIC_PATH)

@route('/hello/<name>')
def index(name):
  return template('<b>Hello {{name}}</b>!', name=name)

parser = argparse.ArgumentParser(description='Running air server')
parser.add_argument('--root_path', type=str, help='filepath to server root')
args = parser.parse_args()

run(reloader=True, host='localhost', port=8080)
