import numpy
import cv2
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import math


# function that applies a color conversion to an
# entire image. It doesn't affect the original
# image array put in the function.
# image_target: 3d array representing an image.
# assumes the pixel value is nested at each x, y position
# as a unit (such as [r, g, b])
# f: color channel converter function.
# return: image with a new color space
def apply_color_convert(f, image_target):
    image = numpy.copy(image_target)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            image[y][x] = f(image[y][x])
    return image


# convert an rgb color channel
# to hue, intensity, saturation (HIS)
def rgb_to_his(channel):
    r = channel[0] / 255
    g = channel[1] / 255
    b = channel[2] / 255

    intensity = (r + g + b) / 3
    if (intensity != 0):
        saturation = 1 - (min(r, g, b) / intensity)
    else:
        saturation = 0

    # calculating the hue is painful
    numerator = (r - g) + (r - b)
    denominator = 2 * math.sqrt((r - g) ** 2 + (r - b) * (g - b))

    theta = 0
    if (denominator != 0):
        theta = math.acos(numerator / denominator)
    else:
        theta = 0

    H = 0
    if b <= g:
        H = theta

    if b > g:
        H = 2 * math.pi - theta

    hue = (H * 180) / math.pi

    return [hue % 360, intensity * 255, saturation * 255]

def convert_to_grayscale(image_array):
    grayscale_image = numpy.zeros((image_array.shape[0], image_array.shape[1]), numpy.uint8)
    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            c = image_array[y][x]
            grayscale_image[y][x] = int(0.299 * c[2] + 0.587 * c[1] + 0.114 * c[0])
    return grayscale_image

def hsv_to_hsl(channel):
    h = channel[0]
    s = float(channel[1]) / float(255)
    v = float(channel[2]) / float(255)

    lightness = v * (1-s/2)
    if lightness == 0 or lightness == 1:
        saturation = 0
    else:
        saturation = (v - lightness) / min(lightness, 1 - lightness)

    return [h, int(saturation * 255), int(lightness * 255)]

# converting from HSI back to RGB

def hsi_to_rgb(channel):
    h = channel[0]
    s = channel[1] / 255
    i = channel[2] / 255

    radians = numpy.deg2rad(h)

    if h < 120:  # red-green sector
        b = i * (1 - s)
        r = i * (1 + (s * numpy.cos(radians) / numpy.cos(numpy.pi / 3 - radians)))
        g = 3 * i - (r + b)
    elif h < 240:  # green-blue sector
        radians -= numpy.deg2rad(120)
        r = i * (1 - s)
        g = i * (1 + (s * numpy.cos(radians) / numpy.cos(numpy.pi / 3 - radians)))
        b = 3 * i - (r + g)
    else:  # blue-red sector
        radians -= numpy.deg2rad(240)
        g = i * (1 - s)
        b = i * (1 + (s * numpy.cos(radians) / numpy.cos(numpy.pi / 3 - radians)))
        r = 3 * i - (g + b)
    return [r * 255, g * 255, b * 255]