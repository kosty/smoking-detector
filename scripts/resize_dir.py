#!/usr/bin/python
from PIL import Image
from pathlib import *
import os, sys

def resize(item, side_len_px=200):
    if item.is_file() and item.suffix != "" and item.suffix != ".svg":
        im = Image.open(item)
        par = item.parent
        target = par.parent/(par.name+"_resized")
        target.mkdir(parents=True, exist_ok=True)
        dest = target/(item.stem+'.jpg')
        imResize = im.resize((side_len_px, side_len_px), Image.LANCZOS)
        imResize = imResize.convert("RGB")
        imResize.save(dest, 'JPEG', quality=90)


if __name__ == "__main__":
    path = Path(sys.args[1])
    assert(path.is_dir())
    # dirs = os.listdir( path )
    for item in path.iterdir():
        resize(item)