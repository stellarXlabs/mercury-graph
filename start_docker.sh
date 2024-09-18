#!/bin/bash

docker run -it --rm --name mercury-graph -p 9876:9876 -v $(pwd):/usr/mercury_graph mercury-graph
