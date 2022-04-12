
import pickle
import os
import dill

class DataManager:
    def __init__(self, data_path, format_="pickle") -> None:
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        self.dir = os.listdir(data_path)
        self.format_ = format_

    def is_exist(self, name):
        return (name in self.dir)
    
    def save_data(self, name, obj):
        if self.format_ == "pickle":
            with open(self.data_path+name, "wb") as f:
                pickle.dump(obj, f)
            print("success save : {0}{1}".format(self.data_path, name))
        elif self.format_ == "dill":
            with open(self.data_path+name, "wb") as f:
                dill.dump(obj, f)
            print("success save : {0}{1}".format(self.data_path, name))

    def load_data(self, name):
        if self.format_ == "pickle":
            with open(self.data_path+name, "rb") as f:
                obj = pickle.load(f)
            print("success load : {0}{1}".format(self.data_path, name))
        elif self.format_ == "dill":
            with open(self.data_path+name, "rb") as f:
                obj = dill.load(f)
            print("success load : {0}{1}".format(self.data_path, name))
        return obj