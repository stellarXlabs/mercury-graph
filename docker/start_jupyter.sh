#!/bin/bash

cd tutorials

jupyter labextension disable "@jupyterlab/apputils-extension:announcements"
jupyter lab --ip=0.0.0.0 --port=9876 --allow-root --NotebookApp.token='' --NotebookApp.password=''
