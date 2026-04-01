import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
import PIL.Image as Image
from numpy.random import RandomState
import scipy.io as sio
import PIL
from PIL import Image
from .build_gemotry import initialization, build_gemotry
import torchvision.transforms as transforms


param = initialization()
ray_trafo = build_gemotry(param)

def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data.astype(np.float32)
    data = data*255.0
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

# Funcion para abrir el csv con los directorios de las imagenes
def open_csv(file_path):
    # Las columnas estan en el orden: imagen_sm, sinograma_sm, imagen_cm, sinograma_cm, mascara_metal
    with open(file_path, 'r') as f:
        # Extraemos la primera linea para obtener los nombres de las columnas
        columns = f.readline().strip().split(',')
        # Creamos un diccionario para almacenar los indices de las columnas
        column_indices = {name: index for index, name in enumerate(columns)}
        # Leemos el resto del archivo y almacenamos los datos en una lista de listas
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(line.strip().split(','))
        # Pasamos las diferentes listas a un diccionario para facilitar el acceso a los datos
    
    data_dict = {name: [row[index] for row in data] for name, index in column_indices.items()}
    return data_dict

# Funcion para leer las imagenes en .raw
def read_raw_image(file_path, width, height, dtype=np.float32):
    pass

class MARTrainDataset(udata.Dataset):
    def __init__(self, dir, transform = None, mask = None):
        super().__init__()
        self.dir = dir # Esto tiene que ser el csv
        self.train_mask = mask
        self.transform = transform  
        self.txtdir = open_csv(self.dir) # Aqui va el directorio de tu csv
        self.file_num = len(self.txtdir['image_sm']) # Esto nos da el numero de archivos
        self.rand_state = RandomState(66)
    
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        # Conseguir los directorios de las imagenes a partir del csv
        img_sm_dir = self.txtdir['image_sm'][idx]
        img_cm_dir = self.txtdir['image_cm'][idx]
        sino_sm_dir = self.txtdir['sinogram_sm'][idx]
        sino_cm_dir = self.txtdir['sinogram_cm'][idx]
        mask_dir = self.txtdir['mascara_metal'][idx]

        # Cargar las imagenes a partir de los directorios, utilizando tu funcion read_raw_image, y especificando el tamaño de las imagenes (512x512 para las imagenes y 512x416 para los sinogramas)
        img_sm = read_raw_image(img_sm_dir, 512, 512)
        img_cm = read_raw_image(img_cm_dir, 512, 512)
        sino_sm = read_raw_image(sino_sm_dir, 512, 416)
        sino_cm = read_raw_image(sino_cm_dir, 512, 416)
        mask = read_raw_image(mask_dir, 512, 512)

        # Transformar las imagenes para utilizarlas en el entrenamiento
        if self.transform:
            img_sm = self.transform(img_sm)
            img_cm = self.transform(img_cm)
            sino_sm = self.transform(sino_sm)
            sino_cm = self.transform(sino_cm)
            mask = self.transform(mask)
        
        return img_sm, sino_sm, img_cm, sino_cm, mask
    
if __name__ == "__main__":
    archivo_csv = "ruta/al/archivo.csv" # Aqui va el directorio de tu csv
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = MARTrainDataset(archivo_csv, transform=train_transform)
    data = dataset[0] # Esto nos da el primer elemento del dataset, que es una tupla con las imagenes y la mascara
    print(data[0].shape) # Esto nos da la forma de la imagen sm, que deberia ser (1, 512, 512) despues de la transformacion
    print(data[1].shape) # Esto nos da la forma del sinograma sm, que deberia ser (1, 512, 416) despues de la transformacion
    print(data[2].shape) # Esto nos da la forma de la imagen cm, que deberia ser (1, 512, 512) despues de la transformacion
    print(data[3].shape) # Esto nos da la forma del sinograma cm, que deberia ser (1, 512, 416) despues de la transformacion
    print(data[4].shape) # Esto nos da la forma de la mascara, que deberia ser (1, 512, 512) despues de la transformacion
    