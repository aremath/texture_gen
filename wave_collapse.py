import itertools
import random
from collections import Counter
from collections import defaultdict
from PIL import Image
import numpy as np
import argparse
import functools
import math
# Wavefunction collapse for room generation
# inspiration from : https://github.com/mxgmn/WaveFunctionCollapse
# Each "pixel" is in a superposition of possible pixel values
# In this case, the possible values are indexes into the physical tile table
# as well as the tile palette table.
#
# We first break the input up into n x n tiles, then match the tiles with the
# input, which might be completely blank.
# For each pixel, we store what tiles match in the area surrounding it, and
# the values that the pixel would have if each of those tiles was actually at
# that location.
# The result is a weighted probability distribution over the possible
# pixel values for that location. We then choose to fill the
# *most constrained* pixel (the one with the lowest entropy) by choosing a value from its probability
# distribution. Repeating this process eventually produces a value
# for every pixel (or perhaps leaving a few pixels with no matching tiles)

#INPUTS:
# Input image I (n+1-dimensional np array: n of k objects)
# Tile size t (n-tuple - nonzero)
# Initial output image O (n+1-array: n of k objects)

#WHILE COMPUTING:
# Matches M (n+1-array of size O: k * (counter, total))
# Subs S (n-array of size O: set(tiles that work))
# Tiles T (list of n+1-arrays of size t * k)

def wavefunction_collapse(input_array, output_array, tile_size):
    niters = 0
    #print(output_array)
    #print(input_array)
    print("Tile Size: {}".format(tile_size))
    tiles = find_tiles(input_array, tile_size)
    print("Found {} tiles".format(len(tiles)))
    print("Initializing Matches")
    olds, matches, subs = init_matches(output_array, tiles, tile_size)
    while None in output_array:
        print("Iteration: {}".format(niters))
        niters += 1
        to_collapse = find_min_entropy(output_array, matches)
        if to_collapse is not None:
            collapse(matches, output_array, to_collapse)
            print("Collapsed {} to {}".format(to_collapse, output_array[to_collapse]))
            recalc(matches, output_array, subs, to_collapse, tiles, tile_size, olds)
        else:
            print("Done Collapsing")
            break
        # Debug
        #i = copy_replace_none(output_array, (255,255,255))
        #i_out = image_from_1d_array(i)
        #i_out.save("out_{}.png".format(niters))
        
    #TODO use the old values to determine pixel color
    # Use the old nonzero dist to collapse the rest
    #nones = find_none(output_array)
    #for n in nones:
    #    print(olds[n])
    #    collapse(olds, output_array, n)
    return output_array

def image_to_array(image):
    return np.array(image)

def image_to_1d_array(image):
    a = image_to_array(image)
    new_shape = list(a.shape)[:-1] + [1]
    out = np.empty(new_shape, dtype=object)
    for i in np.ndindex(out.shape):
        out[i] = tuple(a[i[0],i[1],:])
    return out

def array_to_image(array):
    a = array.astype(np.uint8)
    return Image.fromarray(a)

def copy_replace_none(array, w):
    out = np.empty(array.shape, dtype=object)
    for i in np.ndindex(array.shape):
        if array[i] is None:
            out[i] = w
        else:
            out[i] = array[i]
    return out

def replace_none(array, w):
    for i in np.ndindex(array.shape):
        if array[i] is None:
            array[i] = w

def find_none(array):
    nones = []
    for i in np.ndindex(array.shape):
        if array[i] is None:
            nones.append(i)
    return nones

def image_from_1d_array(array):
    new_shape = list(array.shape[:-1]) + [3]
    out = np.empty(new_shape)
    for i in np.ndindex(array.shape):
        for j in range(3):
            out[i[0],i[1],j] = array[i][j]
    return array_to_image(out)

# Assumes the last item of the size is k
def add_nothing_boundary(array):
    dims = len(array.shape) - 1
    index_add = [2] * dims + [0]
    new_shape = eltwise_plus(array.shape, index_add)
    out = np.empty(new_shape, dtype=object)
    for index in np.ndindex(out.shape):
        out[index] = Nothing()
    for index in np.ndindex(array.shape):
        new_index = eltwise_plus(index, [1] * dims + [0])
        out[new_index] = array[index]
    return out

def remove_nothing_boundary(array):
    dims = len(array.shape) - 1
    index_sub = [2] * dims + [0]
    new_shape = eltwise_minus(array.shape, index_sub)
    out = np.empty(new_shape, out.shape)
    for new_index in np.ndindex(new_shape):
        old_index = eltwise_plus(new_index, [1]*dims + [0])
        out[new_index] = array[old_index]
    return out

# All locations where a tile of the given size could be placed within the array
# Just make sure the last item of tile_size == k
def tile_placement_iter(array, tile_size):
    dims = range(len(tile_size))
    return itertools.product(*[range(array.shape[i] - tile_size[i] + 1) for i in dims])

# Just make sure the last item of tile_size == k
def find_tiles(input_array, tile_size):
    tiles = []
    for index in tile_placement_iter(input_array, tile_size):
        ixgrid = np.ix_(*[range(index[i], index[i] + size) for i, size in enumerate(tile_size)])
        tiles.append(input_array[ixgrid])
    return tiles

def entropy(match, base=2):
    counter, total = match
    if total == 0:
        return 0
    e = 0
    m = map(lambda x: x/total * math.log(total/x, base), counter.values())
    return sum(m)

ventropy = np.vectorize(entropy)

def find_min_entropy(output_array, matches):
    e = ventropy(matches)
    #print("Average Entropy: {}".format(np.mean(e)))
    # Find the minimum entropy
    min_e = np.inf
    for index in np.ndindex(e.shape):
        # Minimum over only unconstrained tiles
        if output_array[index] is None:
            #print("None at: {}".format(index))
            #print("Matches: {}".format(matches[index]))
            #print("With Entropy: {}".format(e[index]))
            # Want collapsable elements
            if e[index] < min_e and len(matches[index][0]) > 0:
                min_e = e[index]
    # Find all indices for the minimum entropy
    min_is = []
    for index in np.ndindex(e.shape):
        if e[index] == min_e and output_array[index] is None and len(matches[index][0]) > 0:
            min_is.append(index)
    print("{} options for collapse with entropy {}".format(len(min_is), min_e))
    # Only need one -> choose randomly
    if len(min_is) > 0:
        c = random.choice(min_is)
        print("Counter: {}".format(matches[c]))
        return c
    else:
        return None

def set_array(shape):
    s = np.empty(shape, dtype=object)
    for index in np.ndindex(s.shape):
        s[index] = set()
    return s

def match_array(shape):
    s = np.empty(shape, dtype=object)
    for index in np.ndindex(s.shape):
        c = Counter()
        s[index] = [c, 0]
    return s

# Takes the output image (which can be blank)
# and creates a counter for each pixel of what pixel values it could be
def init_matches(output_array, tiles, tile_size):
    subs = set_array(output_array.shape)
    matches = match_array(output_array.shape)
    olds = match_array(output_array.shape)
    for g_index in tile_placement_iter(output_array, tile_size):
        for i,t in enumerate(tiles):
            # If t matches at the given index, update
            if is_match(output_array, matches, t, g_index):
                subs[g_index].add(i)
                for t_index in np.ndindex(t.shape):
                    a_index = eltwise_plus(g_index, t_index)
                    matches[a_index][0][t[t_index]] +=1
                    matches[a_index][1] += 1
                    olds[a_index][0][t[t_index]] +=1
                    olds[a_index][1] += 1
    return olds, matches, subs

# Does tile fit at pos in array?
def is_match(array, matches, tile, pos, check_matches=False):
    for t_index in np.ndindex(tile.shape):
        a_index = eltwise_plus(pos, t_index)
        try:
            # No match if it does not match a collapsed tile
            if array[a_index] is not None:
                if array[a_index] != tile[t_index]:
                    return False
            # No match if it is not a possible choice
            elif check_matches and matches[a_index] is not None:
                if tile[t_index] not in matches[a_index]:
                    return False
        # Index out of bounds is bad too
        except IndexError:
            return False
    return True

def collapse(matches, output_array, index):
    c,t = matches[index]
    assert output_array[index] is None
    collapse_val = random.choices(list(c.keys()), weights=list(c.values()))[0]
    output_array[index] = collapse_val
    matches[index] = (Counter(), 0)

#TODO old matches
def recalc(matches, output_array, subs, index, tiles, tile_size, olds):
    # Calculate the region affected by collapsing the value at index
    # First, calculate the minimum index
    min_index = eltwise_op(index, tile_size, lambda t: max(t[0] - t[1] + 1, 0))
    max_possible = eltwise_minus(matches.shape, tile_size) #TODO can keep
    max_index = [min(min_index[i] + tile_size[i], max_possible[i] + 1) for i in range(len(tile_size))]
    indices = eltwise_minus(max_index, min_index)
    print("Indices:")
    print("Pos: {}".format(index))
    print("Affecting {} through {}".format(min_index, max_index))
    print("For a total of {}".format(indices))
    for t_index in np.ndindex(indices):
        # Find the global index for updating
        g_index = eltwise_plus(min_index, t_index)
        # Update it!
        # For each of the tiles that was placed there, see if it still fits
        to_remove = set()
        assert len(subs[g_index]) > 0, g_index
        for t in subs[g_index]:
            tile = tiles[t]
            # If it is not a match, then we need to remove it
            if not is_match(output_array, matches, tile, g_index):
                # First, remove it from subs (not during iteration)
                to_remove.add(t)
                # Then, remove its influence from the other tiles
                for s_index in np.ndindex(tile_size):
                    gs_index = eltwise_plus(g_index, s_index)
                    #Debug:
                    m_d = matches[gs_index][0]
                    m_t = matches[gs_index][1]
                    assert sum(m_d.values()) == m_t, "Before: {} w {} @ {}".format(m_d, m_t, gs_index)
                    # If the total is zero, then it is already resolved
                    if matches[gs_index][1] != 0:
                        match_d = matches[gs_index][0]
                        match_d[tile[s_index]] -= 1
                        matches[gs_index][1] -= 1
                        if match_d[tile[s_index]] == 0:
                            del match_d[tile[s_index]]
                    #More Debug
                    m_d = matches[gs_index][0]
                    m_t = matches[gs_index][1]
                    assert sum(m_d.values()) == m_t, "After: {} w {} @ {}".format(m_d, m_t, gs_index)
        # Remove the tiles that no longer fit
        print("Initially {} tiles matched".format(len(subs[g_index])))
        subs[g_index] = subs[g_index] - to_remove
        print("Removed {} tiles at {}".format(len(to_remove), g_index))
    #TODO
    # Update olds #TODO: not the right update area
    #for t_index in np.ndindex(indices):
    #    # Find the global index for updating
    #    g_index = eltwise_plus(min_index, t_index)
    #    if matches[g_index][1] != 0:
    #        olds[g_index] = matches[g_index].copy()
    return

# Get iterator over all possible values of indexes into a table of the given size
def multirange(sizes):
    return itertools.product(*[range(i) for i in sizes])

def eltwise_plus(t1, t2):
    return eltwise_op(t1, t2, sum)

def eltwise_minus(t1, t2):
    return eltwise_op(t1, t2, lambda t: t[0] - t[1])

def eltwise_op(t1, t2, f):
    return tuple(map(f, zip(t1, t2)))

def eltwise_match(t1, t2):
    eq = [i1 == i2 for (i1, i2) in zip(t1, t2)]
    return all(eq)

# Special type used for boundaries
class Nothing(object):

    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, Nothing)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig")
    parser.add_argument("--out")
    parser.add_argument("--w")
    parser.add_argument("--h")
    args = parser.parse_args()
    w = int(args.w)
    h = int(args.h)
    shape = (w, h, 3)
    a_out = np.empty(shape, dtype=object)
    i_in = Image.open(args.orig)
    a_in = image_to_array(i_in)
    print(a_in.shape)
    wavefunction_collapse(a_in, a_out, (5,5,3))
    i_out = array_to_image(a_out)
    i_out.save(args.out)

