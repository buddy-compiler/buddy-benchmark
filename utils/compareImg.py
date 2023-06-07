# pip install pillow
from PIL import Image
import numpy as np
import sys

if len(sys.argv) < 3:
    sys.exit("Please provide paths to two images.")

# read two images
img1 = Image.open(sys.argv[1])
img2 = Image.open(sys.argv[2])

img1_dim = len(np.array(img1).shape)
img2_dim = len(np.array(img2).shape)
if img1_dim != img2_dim:
    print("Please provide images of the same type. Currently, one is a grayscale image and the other is a color image.")

# compare picture sizes
if img1.size != img2.size:
    print("Img1.size:[%d, %d]"%(img1.size[0], img1.size[1]), "\t\tImg2.size:[%d, %d]"%(img2.size[0], img2.size[1]))
    sys.exit("Please provide images of the same size.")

# compare each pixel
pixels1 = img1.load()
pixels2 = img2.load()
width, height = img1.size
totalPixels = width * height
diffPixels = 0

if img1_dim == 2:
    for x in range(width):
        for y in range(height):
            # calculate pixel difference
            diff = abs(pixels1[x, y] - pixels2[x, y])
            if diff > 1:  # if difference exceeds 0.1, pixels are considered different
                diffPixels += 1
else :
    for x in range(width):
        for y in range(height):
            # calculate pixel difference
            p1 = pixels1[x, y]
            p2 = pixels2[x, y]
            for c in range(3):  # calculate difference for each RGB channel
                diff = abs(p1[c] - p2[c])
                if diff > 1:  # if difference exceeds 0.1, pixels are considered different
                    diffPixels += 1

if diffPixels > 0:
    Similarity = 1 - (diffPixels / totalPixels)
    print("Different pixels: %d"%(diffPixels), "\tTotal pixels: %d"%(totalPixels))
    print("Similarity: %f"%(Similarity))
else:
    print("The two images are similar.")
