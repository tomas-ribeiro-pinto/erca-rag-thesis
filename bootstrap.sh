#!/bin/sh
export FLASK_APP=./api/index.py
pipenv run flask --debug run -h 0.0.0.0 --port 5003
#flask --app ./api/index.py run --host=0.0.0.0 --port=5003