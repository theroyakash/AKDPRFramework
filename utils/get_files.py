# Downloads files from the web
import requests
from PIL import Image

def download(url):
    result = requests.get(url, allow_redirects=True)
    return result

def downloadImageFromURL(url):
    pass