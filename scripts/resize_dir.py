#!/usr/bin/python
from PIL import Image
from pathlib import *
import os, sys

def resize(item, base_width=200):
    if item.is_file() and item.suffix != "" and item.suffix != ".svg":
        im = Image.open(item)
        wpercent = (base_width/float(im.size[0]))
        hsize = int((float(im.size[1])*float(wpercent)))
        par = item.parent
        target = par.parent/(par.name+"_resized")
        target.mkdir(parents=True, exist_ok=True)
        dest = target/(item.stem+'.jpg')
        imResize = im.resize((base_width, hsize), Image.ANTIALIAS)
        imResize = imResize.convert("RGB")
        imResize.save(dest, 'JPEG', quality=90)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    assert(path.is_dir())
    # dirs = os.listdir( path )
    for item in path.iterdir():
        resize(item)