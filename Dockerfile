FROM gcr.io/google-appengine/python

RUN apt-get update
RUN apt-get install -y --fix-missing \
    python-opencv \
    python3-numpy \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python-dev \
    python-numpy \
    python-protobuf\
    software-properties-common \
    zip

RUN apt-get clean
RUN rm -rf /tmp/* /var/tmp/*

RUN virtualenv /env -p python3.6

# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
ENV APPDIR $PWD

# Copy the application's requirements.txt and run pip to install all
# dependencies into the virtualenv.
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

RUN cd ~/ && \
    mkdir -p ~/dlib && \
    git clone https://github.com/davisking/dlib.git ~/dlib/ && \
    cd dlib && \
    python setup.py install --yes USE_AVX_INSTRUCTIONS && \
    cd $APPDIR

ADD . /app

# Run a WSGI server to serve the application. gunicorn must be declared as
# a dependency in requirements.txt.
CMD gunicorn -b :$PORT main:app
