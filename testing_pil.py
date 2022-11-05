from PIL import Image
import os, glob
import numpy as np

size = 128, 128
scale = 0
for infile in glob.glob("Photos/*.jpg", recursive=True):
    file, ext = os.path.splitext(os.path.basename(infile))
    dir_name = os.path.dirname(infile)
    with Image.open(infile, mode="r", formats=None) as im:
        height, width = im.size
        img_nump = np.zeros(shape=(height, width))
        for h in range(height):
            for w in range(width):
                pixel = im.getpixel((h, w))
                # TODO search for green palet colors
