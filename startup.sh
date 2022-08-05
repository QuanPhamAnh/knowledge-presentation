#!/bin/bash
source antenv/bin/activate
gunicorn --bind=0.0.0.0 --timeout 600 startup:app