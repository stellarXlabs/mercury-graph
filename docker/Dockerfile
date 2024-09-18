# Use the official Python 3.10 image as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /usr/mercury_graph

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir pyspark
RUN pip install numpy
RUN pip install pandas
RUN pip install requests
RUN pip install jupyterlab
RUN pip install networkx[default]

# Make port 9876 available to the world outside this container
EXPOSE 9876

# Define environment variable for Docker image
ENV DOCKERIMAGE mercury-graph

# Define environment variable for PySpark to include graphframes package
ENV PYSPARK_SUBMIT_ARGS "--packages graphframes:graphframes:0.8.2-spark3.0-s_2.12 pyspark-shell"

RUN pip install graphframes

# Override the default command to start a bash shell
CMD ["/bin/bash"]
