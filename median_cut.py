# use median cuts for image partition
from PIL import Image
from pixel import *
from functools import reduce

def max_range(pixels):
    """Find the index of the value (r,g,b) along which
    pixels has the maximum range"""
    r = [p[0] for p in pixels]
    g = [p[1] for p in pixels]
    b = [p[2] for p in pixels]
    r_range = max(r) - min(r)
    g_range = max(g) - min(g)
    b_range = max(b) - min(b)
    ranges = [r_range, g_range, b_range]
    max_range = max(ranges)
    for i, m_range in enumerate(ranges):
        if m_range == max_range:
            return i

# given a counter: pixel_value -> int,
# return two new counters which are cuts of the old counter
# along the median of the greatest range of the colos in the old counter
def cut(pixel_count):
    # pick i is the color along which pixel_count has the largest range
    i = max_range(list(pixel_count.keys()))
    # sort the pixels by their i-value
    spixels = sorted(list(pixel_count.keys()), key=lambda x: x[i])
    cut1 = collections.Counter()
    cut2 = collections.Counter()
    n = len(spixels)
    for i,p in enumerate(spixels):
        if i < n/2:
            cut1[p] = pixel_count[p]
        else:
            cut2[p] = pixel_count[p]
    return [cut1, cut2]

def cuts(pixel_count, n):
    current = [pixel_count]
    for i in range(n):
        final = []
        current = [cut(x) for x in current]
        for c in current:
            final.extend(c)
        current = final
    return current

def partition(pixels, cuts):
    """with the pixels from image and the cuts made of that image,
       produce a map from pixel to cut"""
    colors = [int_centroid(cut) for cut in cuts]
    pixel_map = {}
    for p in pixels:
        for i, c in enumerate(cuts):
            if p in c:
                pixel_map[p] = colors[i]
                break
    return pixel_map

# will partition the image into 2^(n_bins) colors
def median_cut(filename, out_filename, n_iters):
    im = Image.open(filename)
    pixel_count = count_pixels(im)
    print("Cutting")
    cs = cuts(pixel_count, n_iters)
    print("Partitioning")
    part = partition(pixel_count.keys(), cs)
    print("Mapping")
    out = image_copy_transform(im, lambda x: part[x])
    out.save(out_filename)
    print("Done")

if __name__ == "__main__":
    median_cut("scrippsart_small.jpg", "median.png", 3)
