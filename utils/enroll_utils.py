import os
from PIL import Image


def image_validation(images:dict)->bool:
    return None not in images.values()

def creat_user_dir(base_path : str,username:str) ->str:
    path = os.path.join(base_path, username)
    os.makedirs(path, exist_ok=True)
    return path

def save_images(images:dict,user_dir:str):
    for angle,img in images.items():
        image = Image.open(img)
        image.save(os.path.join(user_dir,f"{angle}.jpg"))
