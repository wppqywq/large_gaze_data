import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from constants import *


def draw_target(im_: Image, loc_: tuple, shape_: str = 'plus', transparency: float = 0.80, target_size: int = 18):
    # Make an empty image onto which we can draw
    target = Image.new(mode='RGBA', size=im_.size)
    draw = ImageDraw.Draw(target, mode='RGBA')

    # Draw an X
    if shape_ == 'x':
        half = target_size / 2
        width_div = 3
        width = int(target_size / width_div)

        # Draw two diagonal lines
        draw.line(
            (loc_[0] - half, loc_[1] - half, loc_[0] + half, loc_[1] + half),
            width=width,
            fill=(255, 255, 255, int(255 - (255 * transparency))),
        )
        draw.line(
            (loc_[0] - half, loc_[1] + half, loc_[0] + half, loc_[1] - half),
            width=width,
            fill=(255, 255, 255, int(255 - (255 * transparency))),
        )

    # Draw a +. Note the slightly different scaling parameters to ensure that the x and + are equally large
    elif shape_ == 'plus':
        half = target_size / 1.65
        width_div = 4
        width = int(target_size / width_div)

        # Draw two cardinal rectangles
        draw.rectangle(
            (loc_[0] - (width / 2), loc_[1] - half - (width / 2), loc_[0] + (width / 2), loc_[1] + half + (width / 2)),
            fill=(255, 255, 255, int(255 - (255 * transparency))),
        )
        draw.rectangle(
            (loc_[0] - half - (width / 2), loc_[1] - (width / 2), loc_[0] + half + (width / 2), loc_[1] + (width / 2)),
            fill=(255, 255, 255, int(255 - (255 * transparency))),
        )

    # Print sum of pixels in the first RGBA layer to check if the sum is the same
    targ_arr = np.asarray(target)
    print(shape_, np.nansum(targ_arr[:, :, 1]), 'pixel sum')

    # Overlay the target image onto the original
    out = Image.alpha_composite(im_, target)

    return out


def get_stimulus_locs_px():
    # Set size of bins and create 6 locations. Equally spaced, based on a previous pilot
    x_bin = RESOLUTION[0] / 17
    y_bin = RESOLUTION[1] / 10

    loc1 = (round(x_bin * 2.5), round(y_bin * 1.9))  # top left
    loc2 = (round(x_bin * 8.5), round(y_bin * 1.9))  # top center
    loc3 = (round(x_bin * 14.5), round(y_bin * 1.9))  # top right
    loc4 = (round(x_bin * 2.5), round(y_bin * 8.2))  # bottom left
    loc5 = (round(x_bin * 8.5), round(y_bin * 8.2))  # bottom center
    loc6 = (round(x_bin * 14.5), round(y_bin * 8.2))  # bottom right

    return loc1, loc2, loc3, loc4, loc5, loc6


def get_stimulus_loc_dict():
    loc1, loc2, loc3, loc4, loc5, loc6 = get_stimulus_locs_px()
    stim_loc_dict = {
        'Top L': loc1,
        'Top C': loc2,
        'Top R': loc3,
        'Bot L': loc4,
        'Bot C': loc5,
        'Bot R': loc6,
    }
    return stim_loc_dict


if __name__ == '__main__':
    loc1, loc2, loc3, loc4, loc5, loc6 = get_stimulus_locs_px()

    # For each shape at each location, load the image, draw the target, and save the image
    for shape in ['plus', 'x']:
        for i, loc in enumerate([loc1, loc2, loc3, loc4, loc5, loc6]):
            im = Image.open(ROOT_DIR / 'image_hd.png').convert('RGBA')
            im = im.resize(RESOLUTION)

            loc = (int(loc[0]), int(loc[1]))

            im = draw_target(im, loc, shape_=shape)
            im.save(ROOT_DIR / 'stimuli' / f'loc{i + 1}_shape_{shape}.png')

            # # Optional plotting
            # f = plt.figure(figsize=(8, 4.5))
            # ax = f.add_axes([0, 0, 1, 1])
            # ax.axis('off')
            # ax.set(xlim=(0, im.size[0]), ylim=(0, im.size[1]))
            # ax.imshow(im, extent=(0, im.size[0], 0, im.size[1]))
            # plt.show()

    # Also save the two target shapes separately. Uncomment the color arg to set a black background.
    im = Image.new(mode='RGBA', size=(50, 50))  # , color=(0, 0, 0))
    im = draw_target(im, (25, 25), shape_='plus', transparency=0.1)
    im.save(ROOT_DIR / 'stimuli' / f'shape_plus.png')

    im = Image.new(mode='RGBA', size=(50, 50))  # , color=(0, 0, 0))
    im = draw_target(im, (25, 25), shape_='x', transparency=0.1)
    im.save(ROOT_DIR / 'stimuli' / f'shape_x.png')
