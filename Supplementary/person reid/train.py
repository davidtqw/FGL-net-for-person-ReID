from __future__ import print_function, division
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
from model import Light
from random_erasing import RandomErasing
from shutil import copyfile

version = torch.__version__
from samplers import RandomIdentitySampler
from lr_scheduler import LRScheduler
from Loss import TripletLoss, CrossEntropyLabelSmooth, CenterLoss
from IPython import embed

######################################################################
# Options
# --------

gpu_ids = '0'
name = 'mymodel'
data_dir = 'F:\Databases\Market\pytorch'
train_all_1 = 'True'
batchsize = 32
erasing_p = 0.5
color_jitter = True
str_ids = gpu_ids.split(',')
gpu_ids = []
if not os.path.exists('./model/%s' % name):
    os.makedirs('./model/%s' % name)
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
transform_train_list = [
    transforms.Resize([256, 128]),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Pad(10),
    transforms.RandomCrop([256, 128]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=erasing_p)]
if color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list)
}

train_all = ''
if train_all_1:
    train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                              sampler=RandomIdentitySampler(image_datasets[x], batchsize, 4),
                                              num_workers=0)  # 8 workers may work faster
               for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time() - since)
######################################################################
# Training the model
# ------------------
y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, triplet, num_epochs):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train'] / batchsize) * 5  # first 5 epoch

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('#' * 30)
        # update learning rate
        lr_scheduler = LRScheduler(base_lr=3e-2, step=[40, 120],
                                   factor=0.1, warmup_epoch=10,
                                   warmup_begin_lr=3e-4)

        lr = lr_scheduler.update(epoch)
        if epoch in range(60, 120):
            lr = 0.0003
        if epoch in range(120, 130):
            lr = 0.00003

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < batchsize:  # skip the last batch
                    continue
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                temp_loss = []
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # x1, x2,  fmap3_af,fmap3_mf,  fmap4_af,fmap4_mf, = model(inputs)
                # x1, x2, x3, fmap3_af, fmap3_mf, fmap4_af, fmap4_mf, fmap4_af2, fmap4_mf2, fmap3_addf1, fmap4_addf1, fmap4_addf21 = model(
                #     inputs)

                x1, fmap3_af, fmap3_mf, fmap3_af2, fmap3_mf2, fmap3_addf1, fmap3_addf2, loss_pos, _ = model(
                    inputs)

                _, preds1 = torch.max(x1.data, 1)
                # _, preds2 = torch.max(x2.data, 1)
                # _, preds3 = torch.max(x3.data, 1)
                #
                loss1 = criterion(x1, labels)
                # loss2 = criterion(x2, labels)
                # loss3 = criterion(x3, labels)
                #
                loss5 = triplet(fmap3_af, labels)[0]
                loss6 = triplet(fmap3_mf, labels)[0]
                loss13 = triplet(fmap3_af2, labels)[0]
                loss14 = triplet(fmap3_mf2, labels)[0]
                # loss7 = triplet(fmap4_af, labels)[0]
                # loss8 = triplet(fmap4_mf, labels)[0]
                # loss9 = triplet(fmap4_af2, labels)[0]
                # loss10 = triplet(fmap4_mf2, labels)[0]
                loss11 = triplet(fmap3_addf1, labels)[0]
                loss12 = triplet(fmap3_addf2, labels)[0]
                # loss12 = triplet(fmap4_addf1, labels)[0]
                # loss13 = triplet(fmap4_addf21, labels)[0]

                #
                Id_loss = []
                Id_loss.append(loss1)
                # Id_loss.append(loss2)
                # Id_loss.append(loss3)
                num_id = len(Id_loss)
                #
                loss = sum(Id_loss) / num_id + (
                            loss5 + loss6 + loss11) / 3

                # backward + optimize only if in training phase
                if epoch < 5 and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss = loss * warm_up

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if int(version[2]) > 3:  # for the new version like 0.4.0 and 0.5.0
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                a = float(torch.sum(preds1 == labels.data))
                # b = float(torch.sum(preds2 == labels.data))
                # c = float(torch.sum(preds3 == labels.data))
                # #
                running_corrects_1 = a
                running_corrects_2 = running_corrects_1
                running_corrects += running_corrects_2

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # 在日志文件中记录每个epoch的精度和loss
            with open('./model/%s/%s.txt' % (name, name), 'a') as acc_file:
                acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, epoch_acc, epoch_loss))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'train':
                last_model_wts = model.state_dict()
                if epoch < 150:
                    if epoch % 10 == 9:
                        save_network(model, epoch)
                    draw_curve(epoch)
                else:
                    # if epoch%2 == 0:
                    save_network(model, epoch)
                    draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
model = Light('XS', len(class_names), False)
# model.load_state_dict(torch.load('model/mymodel/net_0.pth'))

print(model)

print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
net = model
total_params = sum(p.numel() for p in net.parameters())
print('总参数个数：{}'.format(total_params))
total_trainable_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('需训练参数个数：{}'.format(total_trainable_parameters))

if use_gpu:
    model = model.cuda()

triplet = TripletLoss(margin=0.3)
criterion = CrossEntropyLabelSmooth(num_classes=len(class_names))
# Decay LR by a factor of 0.1 every 40 epochs
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
dir_name = os.path.join('./model', name)
if os.path.isdir(dir_name):
    copyfile('./train.py', dir_name + '/train.py')
    copyfile('model.py', dir_name + '/model.py')

# save opts
model = train_model(model, criterion, triplet, num_epochs=130)
