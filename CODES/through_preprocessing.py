import cv2
from preprocessing import *
from glob import glob
import os
from tqdm import tqdm

path="/home/christian/Desktop/UNIV/DEMETER/TRAINING/"
path1="PLANT-CONCRETE/"
path2=path1[0:-1]+"-PREP/"

clases = glob(f'{path+path1}/*')
a=len(path+path1)
for i in range(len(clases)):
    clases[i]=clases[i][a:]

os.mkdir(path+path2)
for i in clases:
    os.mkdir(path+path2+i)

print(clases)
print("\n \n")

for i in tqdm(clases):
    local_path=path+path1+i+"/"
    count = len(glob(local_path+'*.png'))
    print("Acces to folder class "+str(i))
    for k in tqdm(range(count)):
        a = cv2.imread(local_path+str(k)+'.png')
        cv2.imwrite(path+path2+str(i)+"/"+str(k)+'.png',greenbyCOM(a))