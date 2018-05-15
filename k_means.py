from PIL import Image
import collections
import random

#TODO: implement other RGB color measures
def pixel_distance(p1, p2):
    """straight 3-d euclidean distance (assume RGB)"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5

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

def closest_mean(pixel, means):
    l = sorted(means, key=lambda x: weighted_pixel_distance(pixel, x))
    return l[0]

def add_pixels(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2])

def centroid(counter):
    """find the centroid given a counter of key - pixel value, value - number of that pixel"""
    total_n = sum(counter.values())
    pos = (0, 0, 0)
    for p in counter.elements():
        pos = add_pixels(pos, p)
    return (pos[0]/total_n, pos[1]/total_n, pos[2]/total_n)

def mean_distance(means1, means2):
    """find the total distance between two sets of means"""
    assert len(means1) == len(means2)
    dists = [weighted_pixel_distance(means1[i], means2[i]) for i in range(len(means1))]
    return sum(dists)

def choose_means(pixel_count, n_means, threshold=0.01):
    """Given a count of pixels, pick the n most representative"""
    #TODO: normalize the threshold by the number of means?
    # initialize means randomly
    means = random.sample(list(pixel_count.keys()), n_means)
    # key - mean, value - counter of pixels assigned to that mean
    n_iter = 0
    while True:
        print("Iteration {!s}".format(n_iter))
        n_iter = n_iter + 1
        mean_classify = collections.defaultdict(lambda: collections.Counter())
        # classify all points to their appropriate mean
        for p, n in pixel_count.items():
            m = closest_mean(p, means)
            mean_classify[m][p] += n
        # update the means to the centroids of the classifications
        new_means = [centroid(mean_classify[m]) for m in means]
        d = mean_distance(means, new_means)
        means = new_means
        #print(means)
        print(d)
        if d <= threshold:
            break
    print("Done")
    # convert to integer
    return [tuple(map(int, list(m))) for m in means]

def k_means(filename, out_filename, n_means):
    im = Image.open(filename)
    out = Image.new("RGBA", im.size)
    pixel_count = count_pixels(im)
    means = choose_means(pixel_count, n_means, threshold=2)
    pixels = []
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            p = im.getpixel((x, y))
            pixels.append(closest_mean(p, means))
    out.putdata(pixels)
    out.save(out_filename)

if __name__ == "__main__":
    k_means("img.jpg", "ayy.png", 8)

