# Docker for Mercury Graph development environment

This repository contains the necessary files to create a Docker container with a development environment to develop Mercury graph.

## Building the image

Form the directory containing the Dockerfile, run the following command:

```bash
./build.sh
```

That builds a docker image called `mercury-graph`.

The image is based on Python 3.10 and contains all the necessary requirements to run mercury-graph and jupyter lab, but does not install
`mercury-graph`. You will install it with `pip install .` each time you run it to avoid having to uninstall it each time you make a change.

## Running the container

From the project root directory, run the following command:

```bash
./start_docker.sh
```

This runs the docker container using the current directory as the user default in the docker container. This way, you can install
`mercury-graph` by simply doing:

```bash
pip install .
```

Additionally, the port 9876 (both in the docker and locally) is exposed to allow its use for jupyter lab.

For your convenience, you can use the script `./docker/start_jupyter.sh` to start jupyter lab in the docker container. This script contains:

```bash
cd tutorials

jupyter labextension disable "@jupyterlab/apputils-extension:announcements"
jupyter lab --ip=0.0.0.0 --port=9876 --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

Your jupyter lab will be available at `http://127.0.0.1:9876/lab` without security tokens. Make sure you are in a network you trust before
running this command.
