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

def xy_average(im, xys):
    total_n = len(xys)
    pos = (0,0,0)
    for xy in xys:
        if xy[0] >= 0 and xy[0] < im.size[0] and xy[1] >= 0 and xy[1] < im.size[1]:
            pos = add_pixels(pos, im.getpixel(xy))
    return (int(pos[0]/total_n), int(pos[1]/total_n), int(pos[2]/total_n))

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

def in_bounds(xy, im):
    return xy[0] >= 0 and xy[0] < im.size[0] and xy[1] >= 0 and xy[1] < im.size[1]

def get_adj(xy, diag_ok):
    out = [(xy[0]+1, xy[1]), (xy[0],xy[1]+1), (xy[0]-1,xy[1]), (xy[0],xy[1]-1)]
    if diag_ok:
        diag = [(xy[0]+1,xy[1]+1), (xy[0]-1,xy[1]+1), (xy[0]+1,xy[1]-1), (xy[0]-1,xy[1]-1)]
        out = out + diag
    return out

# BFS through an image
def image_bfs(im, start, goal_pred, reach_pred=lambda x: True, diag_ok=True):
    assert in_bounds(start, im)
    q = collections.deque([start])
    finished = set([start])
    offers = {start: start}
    while len(q) > 0:
        pos = q.popleft()
        if goal_pred is not None and goal_pred(pos):
            return pos, offers, finished
        neighbors = get_adj(pos, diag_ok)
        for n in neighbors:
            if in_bounds(n, im) and reach_pred(n) and n not in finished:
                q.append(n)
                finished.add(n)
                offers[n] = pos
    return None, offers, finished

# Return the set of all xy pairs in the image
def im_xys(im):
    xys = set()
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            xys.add((x, y))
    return xys


