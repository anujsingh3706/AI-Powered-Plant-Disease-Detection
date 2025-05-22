#!/bin/bash
cd Flask\ Deployed\ App
gunicorn app:app -c gunicorn_config.py 