from PIL import Image
import numpy as np
import pickle
import os


class Model:
    def __init__(self, img_dir: str):
        self.imgs = [np.array(Image.open(f"{img_dir}/_{i}.png")) for i in range(10)]
    
    def cal_diff(self, pic):
        diff = []
        for index, img in enumerate(self.imgs.copy()):
            diff.append(np.sum((img - pic)**2))
        return diff


class DiffClassifier:
    def __init__(self, model: str = None):
        self.model = None
        if model is not None:
            self.load_model(model)
    
    def build_model(self, path):
        self.model = Model(img_dir=path)
    
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

    def predict(self, pic):
        res = self.model.cal_diff(pic)
        return res.index(min(res))


if __name__ == "__main__":
    # dc = DiffClassifier()
    # dc.build_model("images/standard_nums")
    # dc.save_model("diff_model")

    file = "Cropped_Number_yfzm/59.png"
    dc = DiffClassifier("diff_model")
    res = dc.predict_from_path(file)
    print(res)
