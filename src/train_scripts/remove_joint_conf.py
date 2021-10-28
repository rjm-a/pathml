import json
import yaml
from PIL import Image


def read_json(filename):
    with open(filename) as infile:
        data = json.load(infile)

    return data


def write_json(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def read_yaml(filename):
    config = yaml.load(open(filename), yaml.FullLoader)
    return config


def image_dims(path):
    # returns width, height of an image
    # width, height = img.size
    img = Image.open(path)
    return img.size
