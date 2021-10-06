import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

directory='Output_RGB_img'
masterpath='data'
if not os.path.exists(directory):
    os.makedirs(directory)

def Encode_RGB(masterpath):
    mypath=masterpath

    w, h = 1024,1024 
    w2, h2 = 64,64
    my_list = os.listdir(mypath)
    
    for i in range(len(my_list)):
        Subpath=mypath+'/'+my_list[i]
        imgpath=my_list[i]
        onlyfiles = [f for f in listdir(Subpath) if isfile(join(Subpath, f))]
        for s in range(0,len(onlyfiles)):

            data = np.zeros((h, w,3), dtype=np.uint8)

            x=''
            with open(Subpath + '/' + onlyfiles[s]) as f:
                content = f.readlines()
            content = [x.strip() for x in content] 
            for k in range(1,len(content)):
                x+=content[k]
            i=int(w/2)-1
            j=int(h/2)-1

            for k in range (0,len(x)):
                if x[k]=='a' or x[k]=='A':
                    i-=1
                    j-=1
                    if(i==-1 or i==w or j==-1 or j==h):
                        i=int((w/2)-1)
                        j=int((h/2)-1)
                    data[i,j,0]=255
                if x[k]=='c' or x[k]=='C':
                    i-=1
                    j+=1
                    if(i==-1 or i==w or j==-1 or j==h):
                        i=int((w/2)-1)
                        j=int((h/2)-1)
                    data[i,j,1]=255
                if x[k]=='g' or x[k]=='G':
                    i+=1
                    j-=1
                    if(i==-1 or i==w or j==-1 or j==h):
                        i=int((w/2)-1)
                        j=int((h/2)-1)
                    data[i,j,2]=255
                if x[k]=='t' or x[k]=='T':
                    i+=1
                    j+=1
                    if(i==-1 or i==w or j==-1 or j==h):
                        i=int((w/2)-1)
                        j=int((h/2)-1)
                    data[i,j,1]=255
                    data[i,j,2]=255


            img = Image.fromarray(data)
            new_image = img.resize((w2,h2))

            if not os.path.exists(directory+'/'+imgpath):
                os.makedirs(directory+'/'+imgpath)

            imagepath=directory+'/'+imgpath+'/'+onlyfiles[s]+'.png'
            new_image.save(imagepath)



if __name__ == '__main__':
    Encode_RGB(masterpath)