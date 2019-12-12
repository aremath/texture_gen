# Texture Gen

This repository is a collection of image-processing techniques mostly developed in order to learn about methods and/or test things. As such, most of the files here do not have a functioning CLI.

## Wavefunction Collapse

There are currently two implementations of wavefunction collapse. One is image-based tile-by-tile in `prelim.py`. The other numpy-based and is able to collapse things in many dimensions and not just images. This has both the pixel-by-pixel and tile-by-tile methods, and is in `wavecollapse.py`. These both have the same basic CLI:

    python3 prelim.py --orig <path to input image> --out <path to output file> --w <width of output> --h <height of output> --tw <tile width> --th <tile height>

The `wavecollapse` version requires an additional flag `--alg <pixelwise|tilewise>` which is whether to do a pixelwise or a tilewise collapse. The input should be a `.png` image, although probably other formats will work. The output will be saved as the image format you specify. Check PIL docs for the list of accepted formats. In the case of `prelim`, contradiction pixels will appear as black. For `wavecollapse`, you can specify the contradiction pixels when using as a library, but from the command line these pixels will also be black. The pixelwise implementation is still somewhat buggy.

The `Wave Collapse.ipynb` ipython notebook shows how to use the implementation in `wavecollapse` as a library.

## Edge Detect

Edge detection and various image processing techniques that rely on edge detection. This program does not have a CLI and contains several methods that go directly from input image to output image.

## Dither

Dithering using Bayer matrix. Converts the image to black and white.

## K-Means

Uses k-means clustering to reduce the total number of colors in the image.

## Median Cut

Uses a median cut to reduce the total number of colors in the image

## Pixel

Various methods for interacting with pixel color and images, including BFS and pixel distance metrics.
