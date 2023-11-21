from collections import defaultdict

class Data:
    def __init__(self, file_path):
        self.list_dict = self.read_data_from_txt(file_path)
        
    def read_data_from_txt(self, file_path):
        f = open(file_path, "r")
        lines = f.readlines()
        columns = lines[0].split("\t")
        list_dict = []
        for line in lines[1:]:
            values = line.split("\t")
            sample = {columns[i].strip(): values[i].strip() for i in range(len(columns))}
            list_dict.append(sample)
        return list_dict