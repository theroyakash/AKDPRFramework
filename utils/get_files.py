# Downloads files from the web
import requests
from PIL import Image
import io
import os
import sys

def downloadImageFromURL(url):
    '''
    Downloads Image from any image URL
    To know more about what are image URL visit here: https://bit.ly/what-are-imageurl
    '''
    b = requests.get(url).content
    image = Image.open(io.BytesIO(b))
    image.save("saved.jpg")
    print(f'Image saved at {os.getcwd()}/saved.jpg')

downloadImageFromURL('https://images.unsplash.com/photo-1561731216-c3a4d99437d5')