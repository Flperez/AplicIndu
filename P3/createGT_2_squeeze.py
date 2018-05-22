import argparse,glob,os
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

def writeLabel(path_out,lst_info,mode = None):
    if mode:
        path_label = os.path.join(path_out,mode, "labels")
        fimage = "fimages_"+mode+".txt"
        flabel = "flabel_"+mode+".txt"
    else:
        path_label = os.path.join(path_out,"labels")
        fimage = "fimages.txt"
        flabel = "flabel.txt"

    if not os.path.exists(path_label):
        os.makedirs(path_label)
    for info in lst_info:
        path_label_txt = os.path.join(path_label,info.fname.split('/')[-1]+".txt")
        with open(path_label_txt,'w') as file:
            if info.xmins:
                for idx in range(len(info.xmins)):
                    file.write("defecto 0.00 0 0 %d %d %d %d -1 -1 -1 -1 -1 -1 -1\n"
                           %(info.xmins[idx],info.ymins[idx],info.xmaxs[idx],info.ymaxs[idx]))
            else:
                file.write("nothing 0.00 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n")

        with open(os.path.join(path_out,fimage), 'a') as file:
            file.write("%s"%(info.fname)+".png\n")
        with open(os.path.join(path_out, flabel), 'a') as file:
            file.write("%s\n"%(path_label_txt))




if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_in", required=True,
                    help="a las carpetas con las imagenes png y .reg")
    ap.add_argument("--path_out", required=True,
                    help="path a la carpeta donde se crearan las labels y los archivos txts con las rutas")
    ap.add_argument("--split_data",required=False,type=float,default=0.0,
                    help="Splitting (%) path_in files amount")
    ap.add_argument("--random",action="store_true",
                    help="If type it, list of files would be randomized")
    FLAGS = ap.parse_args()

    lst_info = createlstinfo(FLAGS.path_in)
    if FLAGS.random:
        shuffle(lst_info)

    if FLAGS.split_data>0:
        idx = int(FLAGS.split_data*len(lst_info))
        lst1 = lst_info[idx:]
        lst2 = lst_info[:idx]
        writeLabel(FLAGS.path_out, lst1,"training")
        writeLabel(FLAGS.path_out, lst2,"validation")

    else:
        writeLabel(FLAGS.path_out,lst_info)
