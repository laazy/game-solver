from adb import take_screenshot, simulate_touch
from yfzm_ocr import crop_picture
from diff_classifier import DiffClassifier, Model
from picross_solver import Solver


def get_info(dc: DiffClassifier, pics):
    info = []
    for one_line in pics:
        line = []
        for one_num in one_line:
            nums = [dc.predict_from_img(pic) for pic in one_num]
            num = nums[0] if len(nums) == 1 else nums[0] * 10 + nums[1]
            line.append(num)
        info.append(line)
    return info


total_cnt = 0
def save_pics(pics, dc):
    global total_cnt
    for one_line in pics:
        for one_num in one_line:
            for digit in one_num:
                n = dc.predict_from_img(digit)
                digit.save(f"picross/Cropped_Number_yfzm/{n}_{total_cnt}.png")
                total_cnt += 1


def create_train_set():
    for i in range(7):
        screenshot = f"picross/tmp/train{i}.png"
        row_pics, col_pics = crop_picture(screenshot, 25)
        dc = DiffClassifier("picross/diff_model")
        save_pics(row_pics, dc)
        save_pics(col_pics, dc)


def main():
    screenshot = "picross/tmp/image.png"
    take_screenshot(screenshot)
    row_pics, col_pics = crop_picture(screenshot, 25)

    dc = DiffClassifier("picross/diff_model")
    row_info, col_info = get_info(dc, row_pics), get_info(dc, col_pics)
    # save_pics(row_pics, dc)

    solver = Solver()
    solver.load_puzzle(col_info, row_info)
    solver.dump_puzzle_to_file("picross/puzzles/bug.txt")
    solver.solve()

    simulate_touch(solver.board)

if __name__ == "__main__":
    main()