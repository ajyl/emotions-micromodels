#!/bin/bash
source /home/ajyl/venv_mm/bin/activate

#python -c "import socket as s; sock = s.socket(s.AF_UNIX); sock.bind('/tmp/mm_demo.sock')"

gunicorn -w 4 --bind unix:/tmp/mm_demo.sock emotions.server.app:server --log-level info --error-logfile /home/ajyl/emotions-micromodels/emotions/server/error.log
