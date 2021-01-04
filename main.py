import torch
from vit_pytorch import ViT
from dataset2 import Dataset, TestDataset
import torch.nn as nn
from torch.autograd import Variable
import time as t
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

root_path = '/home/ubuntu/dataset/dfdc_image/train/dfdc_train_part_10/'
test_path = '/home/ubuntu/dataset/dfdc_image/test/'
batch_size = 10 #folder num # image_num = batch_size * 32
num_workers= 4
epoch = 100
dir_name = t.strftime('~%Y%m%d~%H%M%S', t.localtime(t.time()))
log_train = './log/' + dir_name + '/train'
writer = SummaryWriter(log_train)

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.3,
    emb_dropout = 0.3
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(v))
criterion = nn.CrossEntropyLoss()
bce = nn.BCELoss()
sigmoid = nn.Sigmoid()
opt = torch.optim.Adam(v.parameters(), lr=3e-4)
#opt = torch.optim.SGD(v.parameters(), lr=3e-4)
v.cuda()
criterion.cuda()
bce.cuda()
dataloader = torch.utils.data.DataLoader(
    Dataset(root_path), batch_size=batch_size, shuffle=False, num_workers=8
)
test_dataloader = torch.utils.data.DataLoader(
    TestDataset(test_path), batch_size=batch_size, shuffle=False, num_workers=8
)

def plot_confusion_matrix(cm,classes, epoch, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.clf()
    """
    This function prints and plots the confusion matrix very prettily.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    # Specify the tick marks and axis text
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # The data formatting
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Print the text of the matrix, adjusting text colour for display
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_{:03d}.png'.format(epoch))
    #plt.show()


def confident_strategy(pred):
    #pred = np.array(pred)
    #sz = len(pred)
    pred_label = torch.argmax(pred, dim=1)
    #pred = sigmoid(pred.values)
    fakes = torch.count_nonzero(pred_label)
    # 11 frames are detected as fakes with high probability
    if fakes > 11:
        return sigmoid(torch.mean(pred)), 1
    else:
        return sigmoid(torch.mean(pred)), 0
    '''
    if fakes > sz // 2.5 and fakes > 11:
        return torch.mean(pred[pred > t]), 1
    elif torch.count_nonzero(pred < 0.2) > 0.9 * sz:
        return torch.mean(pred[pred < 0.2]), 0
    else:
        return torch.mean(pred), 0
    '''

def confusion_matrix_c(pred, label, TP, FN, FP, TN):
    if pred == label:
        if pred == 1:
            TP +=1
        else:
            TN += 1
    else:
        if pred == 1:
            FP += 1
        else:
            FN += 1

    return TP, FN, FP, TN

best_log_loss = 100
current_epoch_train = 0
current_epoch_test = 0
for i in range(epoch):
    #confusion_matrix = torch.tensor(0, 0, 0, 0).cuda() #TP, FN, FP, TN
    #train

    for j, (image, label) in enumerate(dataloader):
        #train
        #TP, FN, FP, TN = 0, 0, 0, 0
        loss = torch.tensor(0.0).cuda()
        v.zero_grad()
        img = Variable(image.view(-1, 3, 256, 256)).cuda()
        #video_label = label
        label = Variable(label.type(torch.LongTensor)).cuda()
        label = label.view(-1)
        preds = v(img)
        loss = criterion(preds, label)
        '''
        for k in range(preds.shape[0]):
            pred, pred_label = confident_strategy(preds[k])
            loss += criterion(pred, label[k]) #loss += bce(sigmoid(pred), label[k]))
            #TP, FN, FP, TN = confusion_matrix(pred_label, int(label[k].cpu()), TP, FN, FP, TN)
        '''
        loss.backward()
        opt.step()
        if (current_epoch_train)%10 == 0:
            print("{} Train Log Loss: {:.3f}".format(current_epoch_train, loss.data))
            # writer
            writer.add_scalar('train_log_loss', loss.data, current_epoch_train)
            writer.add_images('train_images', image[0], current_epoch_train)
            vutils.save_image(image[0].data, 'train_images.png',normalize=True)
        current_epoch_train += 1

    #test
    all_test_preds = []
    all_labels=[]
    TP_test, FN_test, FP_test, TN_test = 0, 0, 0, 0
    total_test_loss = 0
    for j, (img, label) in enumerate(test_dataloader):
        test_img = img.view(-1, 3, 256, 256).cuda()
        test_label = label.type(torch.FloatTensor).cuda()
        test_preds = v(test_img).view(-1, 32, 2)

        for k in range(test_preds.shape[0]):
            test_pred, test_pred_label = confident_strategy(test_preds[k])
            all_test_preds.append(test_pred_label)
            all_labels.append(int(test_label[k].cpu()))
            total_test_loss += bce(test_pred, test_label[k]).detach() #total_test_loss += bce(sigmoid(test_pred), test_label[k]).detach()
            TP_test, FN_test, FP_test, TN_test = confusion_matrix_c(test_pred_label, int(test_label[k].cpu()), TP_test,
                                                                    FN_test, FP_test, TN_test)
    '''       
    test_loss = criterion(test_preds, test_label)
    test_output = torch.argmax(test_preds, dim=1)
    test_correct = (test_output == test_label).float().sum()
    '''
    if best_log_loss > (total_test_loss/len(test_dataloader)):
        torch.save(v.state_dict(), 'best_model.pt')
        best_log_loss = (total_test_loss/len(test_dataloader))

    test_accuracy = (TP_test + TN_test) / (TP_test + FN_test + FP_test + TN_test + 2e-5)
    test_precision = TP_test / (TP_test + FP_test + 2e-5)
    test_recall = TP_test / (TP_test + FN_test + 2e-5)
    test_f1_score = 2 * ((test_precision * test_recall) / (test_precision + test_recall + 2e-5))
    # writer
    print("{} Test Log Loss: {:.3f}, Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}".format(i+1,
                                                                                                 (total_test_loss/len(test_dataloader)),
                                                                                                 test_accuracy,
                                                                                                 test_precision,
                                                                                                 test_recall))
    writer.add_scalar('test_epoch_log_loss', (total_test_loss/len(test_dataloader)), i + 1)
    writer.add_scalar('test_epoch_accuracy', test_accuracy, i + 1)
    writer.add_scalar('test_epoch_precision', test_precision, i + 1)
    writer.add_scalar('test_epoch_recall', test_recall, i + 1)
    writer.add_scalar('test_epoch_f1score', test_f1_score, i + 1)
    writer.add_scalars('TP,TN,FP,FN' , {'TP' : TP_test, 'TN' : TN_test, 'FP' : FP_test, 'FN' : FN_test}, i+1)
    cm = confusion_matrix(all_labels, all_test_preds)
    plot_confusion_matrix(cm, {"Real:0", "Fake:1"}, i+1)
writer.close()