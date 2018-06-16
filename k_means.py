# Use k-means clustering for image partitioning
from PIL import Image
import collections
import random
from pixel import *

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
    pixel_count = count_pixels(im)
    means = choose_means(pixel_count, n_means, threshold=2)
    out = image_copy_transform(im, lambda x: closest_mean(x, means))
    out.save(out_filename)

if __name__ == "__main__":
    k_means("02.jpg", "ayy.png", 12)

