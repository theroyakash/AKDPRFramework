# Downloads files from the web
import requests
from PIL import Image
import io
import os
import sys
from placeholderfile.generateName import generateName as gn

def downloadImageFromURL(url):
    '''
    Downloads Image from any image URL
    To know more about what are image URL visit here: https://bit.ly/what-are-imageurl
    '''
    b = requests.get(url).content
    image = Image.open(io.BytesIO(b))
    filename = gn(suffix='.jpg', prefix=None, seed=None)
    image.save(filename)
    print(f'Image saved at {os.getcwd()}/{filename}')