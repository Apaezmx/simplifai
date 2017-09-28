import argparse
import bottle
from api import resolve_endpoint
from bottle import error, post, request, route, run, static_file, template, Bottle
from config import config
import os
import tensorflow as tf

app = bottle.Bottle()
plugin = bottle.ext.memcache.MemcachePlugin(servers=['localhost:11211'])
app.install(plugin)

global mem_store

@app.error(404)
def error404(error):
  print error
  return 'Nothing here, sorry'

@app.post('/api')
def serve_api(mc):
  config.set_mc(mc)
  for k, v in request.forms.iteritems():
    print 'Form ' + str(k) + ':' + str(v)
  for k, v in request.files.iteritems():
    print 'File ' + str(k) + ':' + str(v)
  endpoint = request.forms.get('endpoint')
  if not endpoint:
    return 'Endpoint not found'
  response = resolve_endpoint(endpoint, 
          {k: v for k, v in request.forms.iteritems()},
          {k: v for k, v in request.files.iteritems()})
  return response

@app.route('/infer')
def serve_index():
  return static_file('infer.html', root=config.STATIC_PATH)
  
@app.route('/')
def serve_index():
  return static_file('index.html', root=config.STATIC_PATH)

@app.route('/<filepath:path>')
def server_static(filepath):
  return static_file(filepath, root=config.STATIC_PATH)

parser = argparse.ArgumentParser(description='Running air server')
parser.add_argument('--root_path', type=str, help='filepath to server root')
args = parser.parse_args()

config.tf_server = tf.train.Server.create_local_server()
run(app, reloader=True, host='localhost', port=8012, server='cherrypy')
