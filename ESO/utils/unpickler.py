


import io
import torch
import pickle


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else: return super().find_class(module, name)