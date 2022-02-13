#This module define buffer for images. 
import torch
import random 


class BufferImages():
    def __init__(self, depth=50, use_buffer=False):
        self.depth = depth
        self.buffer = []
        self.use_buffer = use_buffer

    def update(self, images):
        if not self.use_buffer:
            return images            
        out = []
        for img in images:
            img = torch.unsqueeze(img.data, 0)        
            if len(self.buffer) < self.depth:
                self.buffer.append(img)
                out.append(img)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.depth - 1)
                    out.append(self.buffer[i].clone()) 
                    self.buffer[i] = img                    
                else:
                    out.append(img)       
        return torch.cat(out, 0)