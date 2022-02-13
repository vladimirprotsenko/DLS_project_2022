#This module define dataset class. 
import glob
import os
from PIL import Image
from torch.utils.data import Dataset


class CycleGanDataset(Dataset):
    def __init__(self, path_to_data, transform=None, data_mode='train'):
        super(CycleGanDataset, self).__init__()
        self.path_to_data = path_to_data
        imgs_list = glob.glob(self.path_to_data + "*")
        self.transform = transform
        self.data_mode = data_mode 
        
        self.data = {}
        self.type_ind_and_len = []

        for type_path in imgs_list:
          type_ind = type_path.split("/")[-1]
          if data_mode in type_ind:
            self.type_ind = os.listdir(type_path)
            self.type_ind_and_len.append([type_ind, len(self.type_ind)])
            self.data[type_ind] = []
            for p in range(len(self.type_ind)):
              path_to_ind = os.path.join(type_path, self.type_ind[p])
              img_for_ind = Image.open(path_to_ind).convert("RGB")
              if self.transform:
                img_for_ind = self.transform(img_for_ind)
              self.data[type_ind].append(img_for_ind) 

    def __len__(self):
        return max([i[1] for i in self.type_ind_and_len])

    def __getitem__(self, indx):
        img_x = self.data[self.type_ind_and_len[0][0]][indx % self.type_ind_and_len[0][1]]
        img_y = self.data[self.type_ind_and_len[1][0]][indx % self.type_ind_and_len[1][1]]
        return img_x, img_y