from PIL import Image
import os, glob
import numpy as np


def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def Is_Green(hex_color):
    # TODO Get the bounds of green in hex
    return True


scale = 0
pixel_amounts = []
count = 0
for infile in glob.glob("Photos/*.jpg", recursive=True):
    scale = input("Please enter the scale: ")
    pixel_amounts.append(0)
    file, ext = os.path.splitext(os.path.basename(infile))
    dir_name = os.path.dirname(infile)
    with Image.open(infile, mode="r", formats=None) as im:
        height, width = im.size
        img_nump = np.zeros(shape=(height, width))
        for h in range(height):
            for w in range(width):
                rgb_color = im.getpixel((h, w))
                # TODO search for green palet colors
                r, g, b = rgb_color
                hex_color = rgb2hex(r, g, b)
                if Is_Green(hex_color):
                    pixel_amounts[count] += 1
    count += 1
