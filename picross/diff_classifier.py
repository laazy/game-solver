from PIL import Image
from typing import List
import pathlib
import numpy as np
import pickle
import os


class Model:
    def __init__(self, data: List[np.array]):
        self.imgs = data
    
    def cal_diff(self, pic):
        p = pic.copy() // 255
        p = np.array(p, dtype=np.int8)
        diff = []
        for img in self.imgs:
            diff.append(np.sum(img * (2 * p - 1)))
        return diff


class DiffClassifier:
    def __init__(self, model: str = None):
        self.model = None
        if model is not None:
            self.load_model(model)
    
    def build_model(self, path):
        model_data = [None for _ in range(10)]
        pic_cnt = [0 for _ in range(10)]
        pics_path = pathlib.Path(path)
        for pic_path in pics_path.iterdir():
            digit = int(str(pic_path.stem)[0])
            img = np.array(Image.open(str(pic_path)), dtype=np.int32)
            img_data = img // 255
            if model_data[digit] is None:
                model_data[digit] = img_data
            else:
                model_data[digit] += img_data
            pic_cnt[digit] += 1
        for digit in range(10):
            md = model_data[digit]
            md[md == 0] = -1
            md[np.logical_and(md > 0, md < pic_cnt[digit])] = 0
            md[md == pic_cnt[digit]] = 1
            # print(md)
        self.model = Model(data=[np.array(d, dtype=np.int8) for d in model_data])

    def load_model(self, file):
        with open(file, "rb") as f:
            self.model = pickle.load(f)
    
    def save_model(self, file):
        if self.model is None:
            print("model doesn't exist, failed")
            return
        with open(file, "wb") as f:
            pickle.dump(self.model, f)
    
    def predict_from_path(self, pic_path):
        target = np.array(Image.open(pic_path))
        return self.predict(target)

    def predict_from_img(self, img):
        target = np.array(img)
        return self.predict(target)

    def predict(self, pic):
        res = self.model.cal_diff(pic)
        return res.index(max(res))


if __name__ == "__main__":
    dc = DiffClassifier()
    dc.build_model("picross/Cropped_Number")
    dc.save_model("picross/diff_model")

    file = "picross/Cropped_Number/6_1311.png"
    # dc = DiffClassifier("picross/diff_model")
    res = dc.predict_from_path(file)
    print(res)
