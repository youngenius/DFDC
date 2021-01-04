import cv2
import os
import glob
from skimage import io
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import torchvision
# dir
#video_root = '/home/hjpark/Emotion-FAN/Data/Train_video/'       # train
video_root = '/home/ubuntu/dataset/dfdc/train/'  #'/home/ubuntu/dataset/dfdc/test/'          # val

#save_root='/home/hjpark/Emotion-FAN/Data/Train_frame/'    # train
save_root='/home/ubuntu/dataset/dfdc_image/train/'#'/home/ubuntu/dataset/dfdc_image/test/'       # val
image_size=256
mtcnn = MTCNN(image_size=image_size, margin=25, device='cuda:0')
mtcnn.cuda()

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor*128) + 127.5
    return np.array(processed_tensor, dtype=np.uint8).transpose(1,2,0)


for path in Path(video_root).rglob('*.mp4'):

    if str(path).split('/')[-2] in ['dfdc_train_part_0', 'dfdc_train_part_10', 'dfdc_train_part_11', 'dfdc_train_part_12', 'dfdc_train_part_13']:
        continue

    video_name = path.name # name of video

    # create dir +str(path).split('/')[-2]
    save_path=os.path.join(save_root, video_name.split('.')[0])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    else:
        continue
    # save frame image
    vidcap = cv2.VideoCapture(str(path))
    success, image = vidcap.read()
    save_image = []
    count=0
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            new_image = mtcnn(image, save_path=save_path+"/frame%d.jpg"%(count))
            count += 1
            #save_image.append(mtcnn(image))
        except:
            print("Except!!")
            #new_image = np.array(torchvision.transforms.ToPILImage()(new_image))
            #plt.imsave(save_path + "/frame%d.jpg" % (count//3), new_image)
            #image = image.numpy().transpose((1, 2, 0))
            #image = inverse_normalize(tensor=image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            #image = np.clip(image, 0, 1)
            #plt.imsave(save_path+"/frame%d.png"%count, image)
            #image = np.array(torchvision.transforms.ToPILImage()(image))
            #cv2.imwrite(save_path+"/frame%d.jpg"%count, image)  # save frame as JPEG file
        success, image = vidcap.read()
    '''
    image_len = len(save_image)
    index_jump = (image_len//8)
    for i, index in enumerate(range(0, image_len, index_jump)):
        out = fixed_image_standardization(save_image[index])
        cv2.imwrite(save_path+"/frame%d.jpg"%(i), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        #plt.imsave(save_path+"/frame%d.jpg"%(i), save_image[index])
    '''



