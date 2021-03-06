# Use edge detection for various things
from PIL import Image
import collections
import random

from .pixel import *

#TODO - normalize threshold to the overall contrast of the image?
#TODO - dynamic threshold based on brightness / contrast measure in each part of the image
# -> Darker parts of the image tend to have a lower contrast, so scale the threshold based on
# the brightness of an nxn neighborhood centered on the pixel.
#TODO - Hue distance vs. Brightness distance - Instead of using a comprehensive distance measure, could
# have a metric purely based on hue or purely based on brightness (convert to an HSL space?), then
# can also do some kind of linear interpolation between the two.
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
    print("Finding Edges")
    edges = edge_detection(im, threshold)
    print("Cleaning Edges")
    edges = clean_edges(im, edges, 6)
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

# Removes connected components of size less than threshold from edges
def clean_edges(im, edges, threshold):
    new_edges = set()
    while len(edges) > 0:
        e = edges.pop()
        #TODO: x in edges might not work...
        # Find the connected component that contains e
        _, _, f = image_bfs(im, e, None, reach_pred = lambda x: x in edges, diag_ok=False)
        #_, _, f = image_bfs(im, e, None, reach_pred = lambda x: x in edges, diag_ok=False)
        if len(f) >= threshold:
            new_edges |= f
        # Regardless of whether we're keeping them, we've processed all the xys
        # in this connected component
        edges -= f
    return new_edges

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

def find_partition(im, edges):
    to_partition = im_xys(im)
    to_partition -= edges
    partitions = []
    while len(to_partition) > 0:
        p = to_partition.pop()
        _,_,f = image_bfs(im, p, None, reach_pred = lambda x: x not in edges)
        partitions.append(f)
        to_partition -= f
    return partitions

def mk_color_dict(im, parts):
    colors = [xy_average(im, p) for p in parts]
    color_dict = {}
    for i, part in enumerate(parts):
        color = colors[i]
        for p in part:
            color_dict[p] = color
    return color_dict

def paint_partitions(im, parts, edges):
    out = Image.new("RGBA", im.size)
    pixels = []
    print("Finding Colors")
    color_dict = mk_color_dict(im, parts)
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            if (x, y) in edges:
                #p = im.getpixel((x,y))
                p = (0,0,0)
            else:
                p = color_dict[(x,y)]
            pixels.append(p)
    out.putdata(pixels)
    return out

def edge_regions(filename, out_filename, edge_threshold, size_threshold):
    im = Image.open(filename)
    print("Finding Edges")
    edges = edge_detection(im, edge_threshold)
    print("Cleaning Edges")
    edges = clean_edges(im, edges, size_threshold)
    print("Finding Partition")
    parts = find_partition(im, edges)
    print("Painting Partition")
    out = paint_partitions(im, parts, edges)
    out.save(out_filename)

if __name__ == "__main__":
    edge_regions("img/Library.jpg", "edge_detect.png", 20, 20)

#TODO:
# Ignore pixels where the search distance is large but the euclidean dist is small
# Cellular automaton

