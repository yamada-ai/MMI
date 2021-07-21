import pickle
import os

class DataManager:
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        self.dir = os.listdir(data_path)

    def is_exist(self, name):
        return (name in self.dir)
    
    def save_data(self, name, obj):
        with open(self.data_path+name, "wb") as f:
            pickle.dump(obj, f)
        print("success save : {0}{1}".format(self.data_path, name))

    def load_data(self, name):
        with open(self.data_path+name, "rb") as f:
            obj = pickle.load(f)
        print("success load : {0}{1}".format(self.data_path, name))
        return obj