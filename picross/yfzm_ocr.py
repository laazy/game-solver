from itertools import product
from typing import List
from PIL import Image
import colorsys
import numpy as np

# from recognize import Recognizer
from diff_classifier import DiffClassifier, Model


def pad_image(image):
    iw, ih = image.size
    w, h = 28, 28
    scale = min(float(w) / float(iw), float(h) / float(ih))
    nw, nh = int(iw * scale), int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    image = image.convert('L').point(lambda x: 0 if x < 140 else 1, mode='1')

    new_image = Image.new('L', (28, 28), (0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def draw_box(px, p1, p2):
    lx, ly = p1
    rx, ry = p2
    for i in range(lx, rx):
        px[i, ly] = (255, 255, 255, 255)
        px[i, ry] = (255, 255, 255, 255)
    for i in range(ly, ry):
        px[lx, i] = (255, 255, 255, 255)
        px[rx, i] = (255, 255, 255, 255)


CONFIG = {
    "25": {
        "side": {
            "start": (0, 626),
            "end": (215, 1478)
        },
        "top": {
            "start": (219, 340),
            "end": (1072, 621)
        }
    }
}

# CONFIG = {
#     "25": {
#         "side": {
#             "start": (0, 500),
#             "end": (215, 1352)
#         },
#         "top": {
#             "start": (215, 225),
#             "end": (1065, 500)
#         }
#     }
# }


class PicInfo:
    def __init__(self, img: Image, x_mid: int, y_mid: int, is_yellow: bool):
        self.img = img
        self.x_mid = x_mid
        self.y_mid = y_mid
        self.is_yellow = is_yellow


def find_area(point, img):
    px = img.load()
    width, height = img.size
    area = set()
    area.add(point)
    board_area = area.copy()
    size = 0
    while len(area) > size:
        size = len(area)
        new_board_area = set()
        for p in board_area:
            p_left = (p[0] - 1, p[1]) if p[0] > 1 else p
            p_right = (p[0] + 1, p[1]) if p[0] < width - 1 else p
            p_up = (p[0], p[1] - 1) if p[1] > 1 else p
            p_down = (p[0], p[1] + 1) if p[1] < height - 1 else p
            if px[p_left] and p_left not in area:
                new_board_area.add(p_left)
                area.add(p_left)
            if px[p_right] and p_right not in area:
                new_board_area.add(p_right)
                area.add(p_right)
            if px[p_up] and p_up not in area:
                new_board_area.add(p_up)
                area.add(p_up)
            if px[p_down] and p_down not in area:
                new_board_area.add(p_down)
                area.add(p_down)
        board_area = new_board_area
    x_min = min(area, key=lambda p: p[0])[0]
    x_max = max(area, key=lambda p: p[0])[0]
    y_min = min(area, key=lambda p: p[1])[1]
    y_max = max(area, key=lambda p: p[1])[1]
    return x_min, y_min, x_max + 1, y_max + 1


def is_pixel_yellow(pixel):
    if len(pixel) == 4:
        pixel = (pixel[0], pixel[1], pixel[2])
    colors = {
        "grey": (127, 127, 127),
        "white": (255, 255, 255),
        "yellow": (255,255,0),
        "light_yellow": (127, 127, 0)
    }
    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])
    distances = {k: manhattan(v, pixel) for k, v in colors.items()}
    color = min(distances, key=distances.get)
    # print(f"debug: {pixel}: {color}")
    return color == "yellow" or color == "light_yellow"


def get_all_pics(origin_pic: Image, row_height: int) -> List[List[PicInfo]]:
    bin_pic = origin_pic.convert('L').point(lambda x: 1 if x > 130 else 0, mode='1')
    # bin_pic.show()
    opx = origin_pic.load()
    px = bin_pic.load()
    x_len, y_len = bin_pic.size
    row_num = y_len // row_height
    all_pics = []
    for y_mid in [int(i * row_height + row_height / 2) for i in range(row_num)]:
        row_pics = []
        x = 0
        while x < x_len:
            if px[x, y_mid]:
                num_box = find_area((x, y_mid), bin_pic)
                # draw_box(opx, (num_box[0], num_box[1]), (num_box[2], num_box[3]))
                row_pics.append(PicInfo(
                    img=pad_image(origin_pic.crop(num_box)),
                    x_mid=(num_box[2] + num_box[0]) // 2,
                    y_mid=y_mid,
                    is_yellow=is_pixel_yellow(opx[(x + 1, y_mid)])  # +1 to skip board
                ))
                x = num_box[2]
            x += 1
        all_pics.append(row_pics)
    return all_pics


def build_side_pics(all_side_pics: List[List[PicInfo]]):
    side_pics = []
    is_two_digits = False
    for row_pics in all_side_pics:
        rows = []
        for row_pic in row_pics:
            if not row_pic.is_yellow:
                assert not is_two_digits
                rows.append([row_pic.img])
            else:
                if is_two_digits:
                    rows[-1].append(row_pic.img)
                    is_two_digits = False
                else:
                    rows.append([row_pic.img])
                    is_two_digits = True
        side_pics.append(rows)
    return side_pics


def build_top_pics(all_top_pics: List[List[PicInfo]], grid_size: int, dim: int):
    def _get_pos(x):
        pos = x // grid_size
        assert pos < dim
        return pos
    top_pics = [[] for _ in range(25)]
    is_two_digits = False
    for row_pics in all_top_pics:
        for row_pic in row_pics:
            col_index = _get_pos(row_pic.x_mid)
            if not row_pic.is_yellow:
                assert not is_two_digits
                top_pics[col_index].append([row_pic.img])
            else:
                if is_two_digits:
                    top_pics[col_index][-1].append(row_pic.img)
                    is_two_digits = False
                else:
                    top_pics[col_index].append([row_pic.img])
                    is_two_digits = True
    return top_pics        


def crop_picture(image_name: str, dim: int):
    config = CONFIG[str(dim)]
    side_box = (*config["side"]["start"], *config["side"]["end"])
    top_box = (*config["top"]["start"], *config["top"]["end"])
    im = Image.open(image_name)
    side = im.crop(side_box)
    top = im.crop(top_box)

    side_y = (side_box[3] - side_box[1]) // dim
    top_x = (top_box[2] - top_box[0]) // dim

    all_side_pics = get_all_pics(side, side_y)
    all_top_pics = get_all_pics(top, 28)

    # side.show()
    # top.show()
    # return None

    return build_side_pics(all_side_pics), build_top_pics(all_top_pics, top_x, dim)


if __name__ == "__main__":
    import pathlib as path
    row_pics, col_pics = crop_picture("picross/tmp/image.png", 25)
    tmp_dir = path.Path("picross/tmp")
    tmp_dir.mkdir(exist_ok=True)
    row_pics_dir = tmp_dir / "rows"
    col_pics_dir = tmp_dir / "cols"
    row_pics_dir.mkdir(exist_ok=True)
    col_pics_dir.mkdir(exist_ok=True)
    for row_num, rows in enumerate(row_pics):
        for col_num, row_pic in enumerate(rows):
            for digit_index, digit_pic in enumerate(row_pic):
                digit_pic.save(f"{str(row_pics_dir)}/{row_num}-{col_num}-{digit_index}.png")
    for row_num, rows in enumerate(col_pics):
        for col_num, row_pic in enumerate(rows):
            for digit_index, digit_pic in enumerate(row_pic):
                digit_pic.save(f"{str(col_pics_dir)}/{row_num}-{col_num}-{digit_index}.png")

