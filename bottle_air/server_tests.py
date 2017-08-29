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
  print 'Serving '
  for k, v in request.forms.iteritems():
    print str(k) + ':' + str(v)
  for k, v in request.files.iteritems():
    print str(k) + ':' + str(v)
  endpoint = request.forms.get('endpoint')
  if not endpoint:
    return 'Endpoint not found'
  response = resolve_endpoint(endpoint, 
          {k: v for k, v in request.forms.iteritems()},
          {k: v for k, v in request.files.iteritems()})
  return response

@route('/infer')
def serve_index():
  return static_file('infer.html', root=config.STATIC_PATH)
  
@route('/')
def serve_index():
  return static_file('index_tests.html', root=config.STATIC_PATH)

@route('/<filepath:path>')
def server_static(filepath):
  return static_file(filepath, root=config.STATIC_PATH)

def start():
  parser = argparse.ArgumentParser(description='Running air server')
  parser.add_argument('--root_path', type=str, help='filepath to server root')
  args = parser.parse_args()

  run(reloader=True, host='localhost', port=8012, server='cherrypy')
start()
