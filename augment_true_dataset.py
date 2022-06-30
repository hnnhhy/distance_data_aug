import glob
import os
import random 
import cv2
import albumentations as A

src_label = "/home/ML/clovaAiSTRtraindata/HKdrivingDistanceLMDBtrainImg/train.txt"
rf = open(src_label, "r")
lb_dict = {}
for l in rf.readlines():
    filename, lb = l.split("\t")
    lb_dict[filename] = lb.rsplit("\n")[0]

rf.close()


##### image augmentation #####

folder_path = "/home/ML/clovaAiSTRtraindata/HKdrivingDistanceLMDBtrainImg/train"
dst_folder = "/home/ML/clovaAiSTRtraindata/HKdrivingDistanceLMDBtrainImg/train_aug"
dst_label = "/home/ML/clovaAiSTRtraindata/HKdrivingDistanceLMDBtrainImg/train_aug.txt"

transform = A.Compose([
    A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.5, hue=0.5, always_apply=True),
    A.ISONoise(p=0.6),
    A.GaussianBlur(blur_limit = 5),
    A.InvertImg()
])

wf = open(dst_label, "w")
for imgpath in glob.glob(os.path.join(folder_path, "*.*")):
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imgbasename = os.path.basename(imgpath)
    imgname = os.path.splitext(imgbasename)[0]
    imglabel = lb_dict[imgbasename]
    wf.write("{}\t{}\n".format(imgbasename, imglabel))

    for i in range(5):
        image_aug = transform(image=image)['image']
        new_file_name =  imgname + "_" + str(i) + ".png" 
        dst_img_path = os.path.join(dst_folder, new_file_name)
        cv2.imwrite(dst_img_path, image_aug)
        wf.write("{}\t{}\n".format(new_file_name, imglabel))

wf.close()