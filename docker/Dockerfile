FROM nvcr.io/nvidia/pytorch:20.01-py3

RUN pip install pandas scikit-learn numpy matplotlib jupyterlab split-folders tqdm cython
RUN apt-get update
RUN apt-get install libyaml-dev -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

EXPOSE 8888
#CMD jupyter lab --allow-root --ip="*" --no-browser