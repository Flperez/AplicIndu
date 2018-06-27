import argparse,glob,os,sys
import cv2
from easydict import EasyDict as edict
from natsort import natsorted
from random import shuffle


def load_reg(path):
    fs = cv2.FileStorage(path,cv2.FILE_STORAGE_READ)
    rect = fs.getNode("rectangles")
    return rect.mat().reshape(-1,4)


def createlstinfo(path_in):
    lst_images = natsorted(glob.glob(path_in+"/*.png"))
    print("Number of images",len(lst_images))
    lst_info = []

    for fname in lst_images:
        config = edict()
        path_name = fname.split('.')[0]
        config.fname = path_name
        reg_path = path_name +".reg"
        if os.path.isfile(reg_path):
            BBs = load_reg(reg_path)
            config.xmins = [BB[0] for BB in BBs]
            config.ymins = [BB[1] for BB in BBs]
            config.xmaxs = [BB[0] + BB[2] for BB in BBs]
            config.ymaxs = [BB[1] + BB[3] for BB in BBs]
        else:
            config.xmins = None
            config.ymins = None
            config.xmaxs = None
            config.ymaxs = None

        lst_info.append(config)
    return lst_info

def writeLabel(path_out,lst_info):

    path_label = os.path.join(path_out, "labels")
    if not os.path.exists(path_label):
        os.mkdir(path_label)
    fimage = "fimages.txt"
    flabel = "flabel.txt"

    if not os.path.exists(path_label):
        print("The path: ",path_label," don't exist")
        os.makedirs(path_label)
        print("Creating")
    for info in lst_info:
        path_label_txt = os.path.join(path_label,info.fname.split('/')[-1]+".txt")
        if info.xmins:
            print("Image: "+info.fname+": "+str(len(info.xmins))+" labels\r")
            with open(path_label_txt,'w') as file:
                for idx in range(len(info.xmins)):
                    file.write("defecto 0.00 0 0 %d %d %d %d -1 -1 -1 -1 -1 -1 -1\n"\
                               %(info.xmins[idx],info.ymins[idx],info.xmaxs[idx],info.ymaxs[idx]))


            with open(os.path.join(path_out,fimage), 'a') as file:
                file.write("%s"%(info.fname)+".png\n")
            with open(os.path.join(path_out, flabel), 'a') as file:
                file.write("%s\n"%(path_label_txt))

    print("Creating ",os.path.join(path_out,fimage))
    print("Creating ",os.path.join(path_out,flabel))




if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_in", required=True,
                    help="a las carpetas con las imagenes png y .reg")
    ap.add_argument("--path_out", required=True,
                    help="path a la carpeta donde se crearan las labels y los archivos txts con las rutas")
    FLAGS = ap.parse_args()

    lst_info = createlstinfo(FLAGS.path_in)
    writeLabel(FLAGS.path_out, lst_info)

