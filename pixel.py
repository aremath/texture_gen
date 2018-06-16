
from PIL import Image
import collections

#TODO: implement other RGB color measures
def pixel_distance(p1, p2):
    """straight 3-d euclidean distance (assume RGB)"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5

#TODO: does the order of operations work here? green seems like it's getting shafted
def weighted_pixel_distance(p1, p2):
    """straight 3-d euclidean distance (assume RGB)"""
    return (2*(p1[0] - p2[0])**2 + 4*(p1[1] - p2[1])**2 + 3*(p1[2] - p2[2])**2)**0.5

def count_pixels(image):
    """return a Counter of key - pixel value"""
    out = collections.Counter()
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            out[image.getpixel((x, y))] += 1
    return out

def add_pixels(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2])

def centroid(counter):
    """find the centroid given a counter of key - pixel value, value - number of that pixel"""
    total_n = sum(counter.values())
    pos = (0, 0, 0)
    for p in counter.elements():
        pos = add_pixels(pos, p)
    return (pos[0]/total_n, pos[1]/total_n, pos[2]/total_n)

def int_centroid(counter):
    """find the centroid given a counter of key - pixel value, value - number of that pixel"""
    total_n = sum(counter.values())
    pos = (0, 0, 0)
    for p in counter.elements():
        pos = add_pixels(pos, p)
    return (int(pos[0]/total_n), int(pos[1]/total_n), int(pos[2]/total_n))

def closest_mean(pixel, means):
    l = sorted(means, key=lambda x: weighted_pixel_distance(pixel, x))
    return l[0]

def image_copy_transform(im, mapping):
    out = Image.new("RGBA", im.size)
    pixels = []
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            p = im.getpixel((x,y))
            pixels.append(mapping(p))
    out.putdata(pixels)
    return out
