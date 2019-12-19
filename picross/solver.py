from adb import take_screenshot, simulate_touch
from yfzm_ocr import crop_picture
from diff_classifier import DiffClassifier, Model
from picross_solver import Solver


def get_info(dc: DiffClassifier, pics):
    info = []
    for one_line in pics:
        line = []
        for one_num in one_line:
            nums = [dc.predict(pic) for pic in one_num]
            num = nums[0] if len(nums) == 1 else nums[0] * 10 + nums[1]
            line.append(num)
        info.append(line)
    return info


def main():
    screenshot = "picross/tmp/image.png"
    take_screenshot(screenshot)
    row_pics, col_pics = crop_picture(screenshot, 25)

    dc = DiffClassifier("picross/diff_model")
    row_info, col_info = get_info(dc, row_pics), get_info(dc, col_pics)

    solver = Solver()
    solver.load_puzzle(col_info, row_info)
    solver.solve()

    simulate_touch(solver.board)

if __name__ == "__main__":
    main()