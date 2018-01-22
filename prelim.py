from PIL import Image, ImageFilter
from collections import defaultdict
import random


def texture_gen(filename, outputFilename="Ayylmao.png", size=(60, 60)):
	texture = Image.open(filename)
	tilesize = (5, 5)
	tiles = generate_tiles(texture, tilesize)
	out = Image.new("RGB", size) #TODO: RGBA? - want to allow black pixels in the texture image

	# key - the (x, y) coordinate in the output image
	# value - the list of compatible tiles
	matches = init_matches(out, tiles)

	# while there are still tiles that match, add them in
	while len(matches.keys()) > 0:
		# aggregate the matches with highest certainty (smallest set of matching tiles)
		min_keys, min_val = min_length_keys(matches)
		#print(str(len(min_keys)) + " options for where to place a tile...")

		# choose one to place into the out image
		min_key = random.choice(min_keys)
		min_tile = random.choice(matches[min_key])
		out.paste(tiles[min_tile], box=min_key)

		#print("chose tile " + str(min_tile) + " for location " + str(min_key) + ". " + str(min_val) + " matches")

		# recalculate matches (ignore black pixels)
		recalc_matches(matches, min_key, tiles, out, tilesize)

		#for key, val in matches.items():
		#	print(len(val))

	out.save(outputFilename)

def recalc_matches(matches, location, tiles, image, tilesize):
	'''Given a set of matching tiles the coordinate of a newly placed tile,
		the image the tile was placed in, and the tile size, revise matches
		so that the set of matching tiles accommodates the newly placed tile'''

	# the change affects all regions which have pixels that overlap the placed tile
	min_x = max(location[0] - (tilesize[0] - 1), 0)
	min_y = max(location[1] - (tilesize[1] - 1), 0)
	max_x = min(location[0] + 2 * tilesize[0] - 1, image.size[0])
	max_y = min(location[1] + 2 * tilesize[1] - 1, image.size[1])

	# update each matches in the range of matches which could have changed
	for x in range(min_x, max_x):
		for y in range(min_y, max_y):
			coord = (x, y)
			new_matches = []
			# first, if the region has been filled, we don't need its matches anymore
			if is_complete(coord, image, tilesize):
				# remove the key if the tile is already set
				matches.pop(coord, None)
				#del matches[coord]
			else:
				# the new list of matching tiles will be a strict subset of the old one
				for tile in matches[coord]:
					# only retain those that still match #TODO: just take the tile with the least error
					if is_match(tiles[tile], coord, image):
						new_matches.append(tile)

				if len(new_matches) > 0:
					matches[coord] = new_matches
				# there are no matching tiles, we can remove this key
				else:
					# remove the key if there are no matches
					matches.pop(coord, None)
					#print("ruled out " + str(coord) + " - no matches")
					#del matches[coord]

def is_complete(location, image, tilesize):
	'''given a location and a tilesize, check if we still need to fill pixels in that region'''
	max_x = min(location[0] + tilesize[0], image.size[0])
	max_y = min(location[1] + tilesize[1], image.size[1])
	for x in range(location[0], max_x):
		for y in range(location[1], max_y):
			coord = (x, y)
			# if there's a single black pixel, the region is not complete
			if image.getpixel(coord) == (0, 0, 0):
				return False
	return True

def is_match(tile, location, image):
	'''given a tile and a coord to place that tile, does it match the given image?'''
	# check if their pixels match. treat black as a wildcard
	min_x = location[0]
	min_y = location[1]
	max_x = min(location[0] + tile.size[0], image.size[0])
	max_y = min(location[1] + tile.size[1], image.size[1])

	for x in range(min_x, max_x):
		for y in range(min_y, max_y):
			coord = (x, y)
			# relative tile coordinates
			tilecoord = (x - min_x, y - min_y)
			# black is wildcard - TODO
			if image.getpixel(coord) != (0, 0, 0):
				# if they don't match, fail its
				if image.getpixel(coord) != tile.getpixel(tilecoord):
					return False
	return True

def min_length_keys(dictionary):
	'''Returns a list of keys for dict whose values have the least length'''
	out = []
	# set the min to the length of the first element
	min_l = len(dictionary[list(dictionary.keys())[0]])
	# find the true minimum
	for key, val in dictionary.items():
		if len(val) < min_l:
			min_l = len(val)
	# construct the list of output keys
	for key, val in dictionary.items():
		if len(val) == min_l:
			out.append(key)
	return out, min_l

def init_matches(out_image, tiles):
	'''Initialize the matches array. To begin, every tile matches in every slot'''
	matches = defaultdict(list)
	tlist = list(tiles.keys())
	for x in range(out_image.size[0]):
		for y in range(out_image.size[1]):
			# every (x, y) coord gets a reference to every tile; every tile matches
			# copying the actual images is memory intensive
			matches[(x, y)] = list(tlist)
	return matches

#TODO: oversamples upper right corner?
def generate_tiles(image, size):
	'''Generate all w x h tiles for a given image, starting at the top left'''
	# key - the (x, y) coordinate of the tile
	# value - the "size" image whose top left corner starts at (x, y)
	tiles = {}
	for x in range(image.size[0] - size[0] + 1):
		for y in range(image.size[1] - size[1] + 1):
			tile = image.crop((x, y, x + size[0], y + size[1]))
			# tile no longer belongs to parent
			tile.load()
			tiles[(x, y)] = tile
	return tiles


if __name__ == "__main__":
	texture_gen("sm_norfair_2.png")