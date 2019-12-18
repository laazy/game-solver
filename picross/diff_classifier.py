from PIL import Image
import numpy as np
import pickle
import os


class Model:
    def __init__(self, img_dir: str = "Standard_Number"):
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
    
    def build_model(self):
        self.model = Model(img_dir="Standard_Number")
    
    def load_model(self, file):
        with open(file, "rb") as f:
            self.model = pickle.load(f)
    
    def save_model(self, file):
        if self.model is None:
            print("model doesn't exist, failed")
            return
        with open(file, "wb") as f:
            pickle.dump(self.model, f)
    
    def predict(self, pic):
        target = np.array(Image.open(pic))
        res = self.model.cal_diff(target)
        return res.index(min(res))


if __name__ == "__main__":
    file = "Cropped_Number_yfzm/59.png"
    dc = DiffClassifier("diff_model")
    res = dc.predict(file)
    print(res)
