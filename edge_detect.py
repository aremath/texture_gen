# Use edge detection for various things
from PIL import Image
import collections
import random
from pixel import *

#TODO - normalize threshold
def edge_detection(im, threshold):
    edges = set([])
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            p = im.getpixel((x, y))
            if y + 1 < im.size[1]:
                p2 = im.getpixel((x, y+1))
                if weighted_pixel_distance(p, p2) > threshold:
                    edges.add((x, y))
            if x + 1 < im.size[0]:
                p2 = im.getpixel((x+1, y))
                if weighted_pixel_distance(p, p2) > threshold:
                    edges.add((x, y))
    return edges

def edge_image(im, threshold):
    edges = edge_detection(im, threshold)
    out = Image.new("RGBA", im.size)
    pixels = []
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            if (x, y) in edges:
                p = (0,0,0,255)
            else:
                p = (255,255,255,255)
            pixels.append(p)
    out.putdata(pixels)
    return out

def edge_detect(filename, out_filename, threshold):
    im = Image.open(filename)
    out = edge_image(im, threshold)
    out.save(out_filename)

def mk_range(m, M, inc):
    if inc < 0:
        return range(M-1, m-1, inc)
    else:
        return range(m, M, inc)

def find_parents(dists, x, y, xi, yi, xmin, xmax, ymin, ymax):
    parents = set([])
    xx = x - xi
    yy = y - yi
    if xmin <= xx and xx < xmax:
        xd, xp = dists[(xx, y)]
    else:
        xd, xp = float("inf"), set([(-1, -1)])
    if ymin <= yy and yy < ymax:
        yd, yp = dists[(x, yy)]
    else:
        yd, yp = float("inf"), set([(-1, -1)])
    # The parent is the one with the smallest d
    mind = min(xd, yd)
    if xd <= mind:
        parents |= (xp)
    if yd <= mind:
        parents |= (yp)
    # The actual distance to xy is the distance to the smaller
    # square, plus 1
    return mind + 1, parents

def find_edge_dists(im, edges, x_i, y_i):
    dists = {}
    xmin = 0
    xmax = im.size[0]
    ymin = 0
    ymax = im.size[1]
    for x in mk_range(xmin, xmax, x_i):
        for y in mk_range(ymin, ymax, y_i):
            if (x, y) in edges:
                parents = set([(x, y)])
                d = 0
            else:
                d, parents = find_parents(dists, x, y, x_i, y_i, xmin, xmax, ymin, ymax)
            dists[(x, y)] = (d, parents)
    return dists

# Merges two distance dictionaries
def merge_min(dist1, dist2):
    out = {}
    for xy in dist1:
        d1, p1 = dist1[xy]
        d2, p2 = dist2[xy]
        if d1 < d2:
            out[xy] = dist1[xy]
        elif d2 < d1:
            out[xy] = dist2[xy]
        # If they're equal, union the parents.
        else:
            out[xy] = (d1, p1 | p2)
    return out

def find_edge_dists_all(im, edges):
    # Find the closest in each of the four diagonal directions
    q1 = find_edge_dists(im, edges, 1, 1)
    q2 = find_edge_dists(im, edges, 1, -1)
    q3 = find_edge_dists(im, edges, -1, 1)
    q4 = find_edge_dists(im, edges, -1, -1)
    # Merge them together for the closest overall
    q12 = merge_min(q1, q2)
    q34 = merge_min(q3, q4)
    qall = merge_min(q12, q34)
    return qall

# Given a set of xy, compute the average color of the xy in the image
# If xy is outside the image, use black...
# TODO: replace with pixel.pixel_average
def average_parents(im, parents):
    total_n = len(parents)
    counter = (0,0,0)
    for xy in parents:
        if xy[0] > 0 and xy[1] > 0:
            counter = add_pixels(counter, im.getpixel(xy))
    return (int(counter[0]/total_n), int(counter[1]/total_n), int(counter[2]/total_n))

# Creates a list of coordinates inside a box whose edges
# are all radius away from the center (using the manhattan distance)
def manhattan_box(center, radius):
    out = []
    yy = range(-radius, radius + 1)
    for y in yy:
        xx = range(abs(y)-radius, radius-abs(y) + 1)
        for x in xx:
            out.append((x + center[0], y + center[1]))
    return out

# Transforms an image so that each pixel is the average color of the closest edges
def closest_edge_transform(im, edges):
    print("Finding Closest Edges")
    dists = find_edge_dists_all(im, edges)
    print("Constructing Image")
    out = Image.new("RGBA", im.size)
    pixels = []
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            p = im.getpixel(dists[(x,y)][1].pop())
            #p = xy_average(im, dists[(x,y)][1])
            pixels.append(p)
    out.putdata(pixels)
    return out

def edge_blur_transform(im, edges, maxblur):
    print("Finding Closest Edges")
    dists = find_edge_dists_all(im, edges)
    print("Constructing Image")
    out = Image.new("RGBA", im.size)
    pixels = []
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            d = min(dists[(x,y)][0], maxblur)
            if (x,y) in edges:
                p = (0,0,0)
            else:
                p = xy_average(im, manhattan_box((x,y), d))
            pixels.append(p)
    out.putdata(pixels)
    return out

def edge_close(filename, out_filename, threshold):
    im = Image.open(filename)
    print("Detecting Edges")
    edges = edge_detection(im, threshold)
    print("Edge transform")
    out = closest_edge_transform(im, edges)
    out.save(out_filename)

def edge_blur(filename, out_filename, threshold, maxblur):
    im = Image.open(filename)
    print("Detecting Edges")
    edges = edge_detection(im, threshold)
    print("Edge transform")
    out = edge_blur_transform(im, edges, maxblur)
    out.save(out_filename)

if __name__ == "__main__":
    edge_blur("charles_1.jpg", "edge_detect.png", 40, 4)

