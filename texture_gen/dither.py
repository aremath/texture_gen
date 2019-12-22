from PIL import Image
import numpy as np

from .pixel import *

def dither_map(palette, pixel_color, pixel_x, pixel_y, dither_matrix, matrix_size):
    #TODO: how to set the ratio?
    matrix_thresh = 8*(dither_matrix[pixel_x % matrix_size][pixel_y % matrix_size] - 0.5)
    #TODO is this right?
    dithered_pixel = add_pixels(pixel_color, (matrix_thresh, matrix_thresh, matrix_thresh))
    return closest_mean(dithered_pixel, palette)

def matrix_add(m1, m2, pos):
    """adds m1 to m2 at pos, assuming m1 and m2 are the same dimension as pos"""
    #NOTE: might be buggy if m1 and m2 alias
    #TODO: might need to reverse the range lists?
    m1_ranges = []
    m2_ranges = []
    for i in range(len(pos)):
        dim = pos[i]
        m1_range = slice(max(0, dim), max(min(dim + m2.shape[i], m1.shape[i]), 0))
        m1_ranges.append(m1_range)
        m2_range = slice(max(0, -dim), min(m1.shape[i] - dim, m2.shape[i]))
        m2_ranges.append(m2_range)
    m1[m1_ranges] += m2[m2_ranges]

#TODO: assert size is a power of 2
def dither_matrix(size):
    """create a size x size dither matrix"""
    if size == 2:
        return (np.array([[0,2],[3,1]]).astype(float)) * 0.25
    else:
        n = size//2
        nsq4 = 4*(n**2)
        mn = dither_matrix(n)
        base = np.zeros((size, size)).astype(float)
        matrix_add(base, nsq4 * mn, (0,0))
        matrix_add(base, nsq4 * mn + 3, (n,0))
        matrix_add(base, nsq4 * mn + 2, (0,n))
        matrix_add(base, nsq4 * mn + 1, (n,n))
        return base * (1./nsq4)

def bw_palette(n):
    palette = []
    for i in range(n+1):
        q = int(float(i)/n * 255) #TODO might have bad rounding errors?
        palette.append((q,q,q))
    return palette

# TODO: these functions should take Image and produce Image
# TODO: for some reason, this is quite slow...
# TODO: can use k-means to determine the palette!
def dither(in_image, palette, matrix_size):
    m =  dither_matrix(matrix_size)
    # Can't use image_copy_transform because the transform depends on xy
    # TODO: allow image_copy_transform to use xy?
    out = Image.new("RGBA", in_image.size)
    pixels = []
    for y in range(in_image.size[1]):
        for x in range(in_image.size[0]):
            p = in_image.getpixel((x,y))
            pixels.append(dither_map(palette, p, x, y, m, matrix_size))
    out.putdata(pixels)
    return out
    
if __name__ == "__main__":
    im = Image.open("scrippsart_median.png")
    #TODO uniform black to white palette function
    out = dither(im, bw_palette(5), 8)
    out.save("dither.png")
