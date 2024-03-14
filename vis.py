# -*- coding: utf-8 -*-            
# Time : 2022/10/5 20:03
# Author : Regulus
# FileName: vis.py
# Explain:
# Software: PyCharm

import os
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from tonic.datasets import NCALTECH101, CIFAR10DVS
from braincog.datasets.datasets import unpack_mix_param, DATA_DIR
from braincog.datasets.cut_mix import *
import tonic
from matplotlib import rcParams
import seaborn as sns
import tonic
from tonic import DiskCachedDataset
import cv2
# from braincog.datasets.datasets import *
# train_loader, test_loader, _, _ = get_cifar10_data(batch_size=10000)
# for (image, label) in test_loader:
#     label = label.float()
#     for i in range(10):
#         num = torch.sum(label == i)
#         print("label: {}, num:{}".format(i, num))

# for matplotlib 3D
def get_proj(self):
    """
     Create the projection matrix from the current viewing position.

     elev stores the elevation angle in the z plane
     azim stores the azimuth angle in the (x, y) plane

     dist is the distance of the eye viewing point from the object point.
    """
    # chosen for similarity with the initial view before gh-8896

    relev, razim = np.pi * self.elev / 180, np.pi * self.azim / 180

    # EDITED TO HAVE SCALED AXIS
    xmin, xmax = np.divide(self.get_xlim3d(), self.pbaspect[0])
    ymin, ymax = np.divide(self.get_ylim3d(), self.pbaspect[1])
    zmin, zmax = np.divide(self.get_zlim3d(), self.pbaspect[2])

    # transform to uniform world coordinates 0-1, 0-1, 0-1
    worldM = proj3d.world_transformation(xmin, xmax,
                                         ymin, ymax,
                                         zmin, zmax)

    # look into the middle of the new coordinates
    R = self.pbaspect / 2

    xp = R[0] + np.cos(razim) * np.cos(relev) * self.dist
    yp = R[1] + np.sin(razim) * np.cos(relev) * self.dist
    zp = R[2] + np.sin(relev) * self.dist
    E = np.array((xp, yp, zp))

    self.eye = E
    self.vvec = R - E
    self.vvec = self.vvec / np.linalg.norm(self.vvec)

    if abs(relev) > np.pi / 2:
        # upside down
        V = np.array((0, 0, -1))
    else:
        V = np.array((0, 0, 1))
    zfront, zback = -self.dist, self.dist

    viewM = proj3d.view_transformation(E, R, V)
    projM = self._projection(zfront, zback)
    M0 = np.dot(viewM, worldM)
    M = np.dot(projM, M0)
    return M



def get_dvsc10_data(batch_size=1, step=10):
    """
    获取DVS CIFAR10数据
    http://journal.frontiersin.org/article/10.3389/fnins.2017.00309/full
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    size = 48
    train_data_ratio = 1.0
    portion = 1.0
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.DropEvent(p=0.0),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])
    train_dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=train_transform)
    test_dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=test_transform)

    # train_transform = transforms.Compose([
    # lambda x: torch.tensor(x, dtype=torch.float),
    # lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    # transforms.RandomCrop(size, padding=size // 12),
    # ])

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])   # 这里lambda返回的是地址, 注意不要用List复用.

    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])   # 这里lambda返回的是地址, 注意不要用List复用.

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(DATA_DIR, 'DVS/DVS_Cifar10/train_cache_{}'.format(step)),
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(DATA_DIR, 'DVS/DVS_Cifar10/test_cache_{}'.format(step)),
                                     transform=test_transform)

    num_train = len(train_dataset)
    num_per_cls = num_train // 10
    indices_train, indices_test = [], []
    for i in range(10):
        indices_train.extend(
            list(range(i * num_per_cls, round(i * num_per_cls + num_per_cls * portion))))
        indices_test.extend(
            list(range(round(i * num_per_cls + num_per_cls * portion), (i + 1) * num_per_cls)))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train),
        pin_memory=True, drop_last=False, num_workers=0
    )  # drop_last应该为True

    return train_loader, None, None, None


def get_dataloader_ncal(step, **kwargs):
    sensor_size = tonic.datasets.NCALTECH101.sensor_size
    transform = tonic.transforms.Compose([
        # tonic.transforms.DropPixel(hot_pixel_frequency=.999),
        tonic.transforms.Denoise(2500),
        # tonic.transforms.DropEvent(p=0.1),
        # tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
        # lambda x: F.interpolate(torch.tensor(x, dtype=torch.float), size=[48, 48], mode='bilinear', align_corners=True),
    ])
    # dataset = tonic.datasets.CEPDVS(os.path.join(DATA_DIR, 'DVS/CEP-DVS'), transform=None)
    dataset = tonic.datasets.NCALTECH101(os.path.join(DATA_DIR, 'DVS/NCALTECH101'), transform=transform)
    # dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=transform)
    # dataset = [dataset[5569], dataset[8196]]
    # dataset = [dataset[5000], dataset[6000]] # 1958
    # dataset = [dataset[0]]
    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, _ = unpack_mix_param(kwargs)
    print(mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise)
    if cut_mix:
        dataset = CutMix(dataset,
                         beta=beta,
                         prob=prob,
                         num_mix=num,
                         num_class=num_classes,
                         vis=True,
                         noise=noise)

    if event_mix:
        dataset = EventMix(dataset,
                           beta=beta,
                           prob=prob,
                           num_mix=num,
                           num_class=num_classes,
                           vis=True,
                           noise=noise,
                           gaussian_n=7)
    if mix_up:
        dataset = MixUp(dataset,
                        beta=beta,
                        prob=prob,
                        num_mix=num,
                        num_class=num_classes,
                        vis=True,
                        noise=noise)

    # loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=1,
    #     shuffle=False,
    #     pin_memory=True, drop_last=True, num_workers=8
    # )
    return dataset


def event_vis_splited(x):
    assert len(x.shape) == 4  # timestep, channel, w, h
    pos_idx1 = []
    neg_idx1 = []
    pos_idx2 = []
    neg_idx2 = []
    for t in range(x.shape[0]):
        for r in range(x.shape[2]):
            for c in range(x.shape[3]):
                if x[t, 0, r, c] > 0:
                    pos_idx1.append((t, r, c))
                if x[t, 1, r, c] > 0:
                    neg_idx1.append((t, r, c))
                if x[t, 0, r, c] < 0:
                    pos_idx2.append((t, r, c))
                if x[t, 1, r, c] < 0:
                    neg_idx2.append((t, r, c))

    if len(pos_idx1) > 0:
        pos_t1, pos_x1, pos_y1 = np.split(np.array(pos_idx1), 3, axis=1)
        neg_t1, neg_x1, neg_y1 = np.split(np.array(neg_idx1), 3, axis=1)
    if len(pos_idx2) > 0:
        pos_t2, pos_x2, pos_y2 = np.split(np.array(pos_idx2), 3, axis=1)
        neg_t2, neg_x2, neg_y2 = np.split(np.array(neg_idx2), 3, axis=1)

    fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
    ax = Axes3D(fig)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.set_zlim(0, 48)
    ax.set_xlabel('t (time step)')
    ax.set_ylabel('w (pixel)')
    ax.set_zlabel('h (pixel)')
    # ax.grid(True)fffffffffff
    # ax.grid(False)
    ax.view_init(elev=15, azim=15)
    # ax.pbaspect = np.array([2.0, 1.0, 0.5])
    # ax.view_init(elev=10, azim=-75)

    if len(pos_idx1) > 0:
        # pos_idx = np.random.choice(pos_t1.shape[0], int(0.1 * pos_t1.shape[0]))
        ax.scatter(pos_t1[:, 0], pos_y1[:, 0], 48 - pos_x1[:, 0], color='red', alpha=0.1, s=2.)
        # pos_idx = np.random.choice(neg_t1.shape[0], int(0.1 * neg_t1.shape[0]))
        ax.scatter(neg_t1[:, 0], neg_y1[:, 0], 48 - neg_x1[:, 0], color='blue', alpha=0.1, s=2.)
    if len(pos_idx2) > 0:
        # pos_idx = np.random.choice(pos_t2.shape[0], int(0.1 * pos_t2.shape[0]))
        ax.scatter(pos_t2[:, 0], pos_y2[:, 0], 48 - pos_x2[:, 0], color='lime', alpha=0.1, s=2.)
        # pos_idx = np.random.choice(neg_t2.shape[0], int(0.1 * neg_t2.shape[0]))
        ax.scatter(neg_t2[:, 0], neg_y2[:, 0], 48 - neg_x2[:, 0], color='fuchsia', alpha=0.1, s=2.)
    return ax


# def event_vis_raw_1d(x):
#     sns.set_style('whitegrid')
#     # sns.set_palette('deep', desat=.6)
#     sns.set_context("notebook", font_scale=1.5,
#                     rc={"lines.linewidth": 2.5})
#
#     x = np.array(x.tolist())  # x, y, t, p   (dvs10 is t, x, y, p)
#     mask = (x[:, 3] == 1)
#     x_pos = x[mask]
#     x_neg = x[mask == False]
#     print(x_pos[:, 1].max())
#     print(x_pos[:, 2].max())
#     # pos_idx = np.random.choice(x_pos.shape[0], 10000)
#     # neg_idx = np.random.choice(x_neg.shape[0], 10000)
#     # x_pos[pos_idx, 2] = 0
#     # x_neg[neg_idx, 2] = 0
#
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, aspect='equal')
#     ax.set_xlabel('w (pixel)')
#     ax.set_ylabel('h (pixel)')
#     # ax.set_xticks([])
#     # ax.set_yticks([])
#     # ax.set_zticks([])
#     # ax.scatter(x_pos[pos_idx, 2], 48 - x_pos[pos_idx, 0], 48 - x_pos[pos_idx, 1], color='red', alpha=0.3, s=1.)
#     # ax.scatter(x_neg[neg_idx, 2], 48 - x_neg[neg_idx, 0], 48 - x_neg[neg_idx, 1], color='blue', alpha=0.3, s=1.)
#     ax.scatter(x_pos[:, 1], x_pos[:, 2], color='red', alpha=0.6, s=4)
#     ax.scatter(x_neg[:, 1], x_neg[:, 2], color='blue', alpha=0.6, s=4)
#     # ax.scatter(x_pos[:, 1], x_pos[:, 2], color='red', alpha=0.6, s=4)

max_x = 0
max_y = 0
def event_vis_raw_1d(x):
    # global max_x, max_y
    # sns.set_style('whitegrid')
    # # sns.set_palette('deep', desat=.6)
    # sns.set_context("notebook", font_scale=1.5,
    #                 rc={"lines.linewidth": 2.5})
    #
    # x = np.array(x.tolist())  # x, y, t, p (dvs10 is t, x, y, p)
    # mask = (x[:, 3] == 1)
    # x_pos = x[mask]
    # x_neg = x[mask == False]
    # # print(x_pos[:, 0].max())
    # # print(x_pos[:, 1].max())
    # if x_pos[:, 0].max() > max_x:
    #     max_x = x_pos[:, 0].max()
    # if x_pos[:, 1].max() > max_y:
    #     max_y = x_pos[:, 1].max()
    # # print(x_pos[:, 2].max())
    # # pos_idx = np.random.choice(x_pos.shape[0], 10000)
    # # neg_idx = np.random.choice(x_neg.shape[0], 10000)
    # # x_pos[pos_idx, 2] = 0
    # # x_neg[neg_idx, 2] = 0
    #
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, aspect='equal')
    # ax.set_xlabel('w (pixel)')
    # ax.set_ylabel('h (pixel)')
    # # ax.set_xticks([])
    # # ax.set_yticks([])
    # # ax.set_zticks([])
    # # ------------------------适合dvsc10---------------------------------------
    # # ax.scatter(48 - x_pos[:, 1] * 0.375, 48 - x_pos[:, 2] * 0.375, color='red', alpha=0.6, s=4)
    # # ax.scatter(48 - x_neg[:, 1] * 0.375, 48 - x_neg[:, 2] * 0.375, color='blue', alpha=0.6, s=4)
    #
    # ax.scatter(48 - x_pos[:, 0] * 0.28, 48 - x_pos[:, 1] * 0.28, color='red', alpha=0.6, s=4)
    # ax.scatter(48 - x_neg[:, 0] * 0.28, 48 - x_neg[:, 1] * 0.28, color='blue', alpha=0.6, s=4)
    #
    # # ax.scatter(239 - x_pos[:, 0] // 2, 179 - x_pos[:, 1], color='red', alpha=0.6, s=4)
    # # ax.scatter(239 - x_neg[:, 0] // 2, 179 - x_neg[:, 1], color='blue', alpha=0.6, s=4)

    x = np.array(x.tolist())  # x, y, t, p (dvs10 is t, x, y, p)
    mask = (x[:, 3] == 1)
    x_pos = x[mask]
    x_neg = x[mask == False]
    x_zoom_ratio = 48.0 / float(x_pos[:, 0].max())
    y_zoom_ratio = 48.0 / float(x_pos[:, 1].max())
    plt.scatter(x_pos[:, 0] * x_zoom_ratio, 48 - x_pos[:, 1] * y_zoom_ratio, color='red', alpha=0.6, s=4)
    plt.scatter(x_neg[:, 0] * x_zoom_ratio, 48 - x_neg[:, 1] * y_zoom_ratio, color='blue', alpha=0.6, s=4)
    plt.savefig('fig/step/all.jpg', dpi=300)

def event_vis_raw_t(x):
    x = np.array(x.tolist())  # x, y, t, p (dvs10 is t, x, y, p)
    mask = (x[:, 3] == 1)
    x_pos = x[mask]
    x_neg = x[mask == False]
    x_zoom_ratio = 48.0 / float(x_pos[:, 0].max())
    y_zoom_ratio = 48.0 / float(x_pos[:, 1].max())
    x_pos_list = np.array_split(x_pos, 10)
    x_neg_list = np.array_split(x_neg, 10)
    for i, (x_pos_t, x_neg_t) in enumerate(zip(x_pos_list, x_neg_list)):
        plt.figure(figsize=(8, 6))
        plt.axis('off')
        plt.scatter(x_pos_t[:, 0] * x_zoom_ratio, 48 - x_pos_t[:, 1] * y_zoom_ratio, color='red', alpha=0.6, s=4)
        plt.scatter(x_neg_t[:, 0] * x_zoom_ratio, 48 - x_neg_t[:, 1] * y_zoom_ratio, color='blue', alpha=0.6, s=4)
        plt.savefig('fig/step/{}.jpg'.format(i), dpi=300)
        plt.close()

def event_vis_raw(x):
    sns.set_style('whitegrid')
    # sns.set_palette('deep', desat=.6)
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    Axes3D.get_proj = get_proj
    x = np.array(x.tolist())  # x, y, t, p
    mask = (x[:, 3] == 1)
    x_pos = x[mask]
    x_neg = x[mask == False]
    pos_idx = np.random.choice(x_pos.shape[0], 10000)
    neg_idx = np.random.choice(x_neg.shape[0], 10000)
    # x_pos[pos_idx, 2] = 0
    # x_neg[neg_idx, 2] = 0

    fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
    ax = Axes3D(fig)
    ax.pbaspect = np.array([2.0, 1.0, 0.5])
    ax.view_init(elev=10, azim=-75)
    ax.set_yticks([60, 90, 120])
    ax.set_zticks([-75, 0, 50])
    # ax.axis('off')
    ax.xaxis.labelpad = 18
    ax.yaxis.labelpad = 5
    ax.set_xlabel('time step')
    ax.set_ylabel('weight')
    ax.set_zlabel('height')
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.scatter(x_pos[pos_idx, 2], 48 - x_pos[pos_idx, 0], 48 - x_pos[pos_idx, 1], color='red', alpha=0.3, s=1.)
    # ax.scatter(x_neg[neg_idx, 2], 48 - x_neg[neg_idx, 0], 48 - x_neg[neg_idx, 1], color='blue', alpha=0.3, s=1.)
    ax.scatter(x_pos[:, 2], 120 - x_pos[:, 0] // 2, 80 - x_pos[:, 1], color='red', alpha=0.3, s=1.)
    ax.scatter(x_neg[:, 2], 120 - x_neg[:, 0] // 2, 80 - x_neg[:, 1], color='blue', alpha=0.3, s=1.)
    ax.scatter(18000, 120 - x_pos[:, 0] // 2, 80 - x_pos[:, 1], color='red', alpha=0.3, s=1.)
    ax.scatter(18000, 120 - x_pos[:, 0] // 2, 80 - x_pos[:, 1], color='blue', alpha=0.3, s=1.)
    plt.savefig('fig/111.jpg', dpi=300)

# def event_vis_raw(x):
#     sns.set_style('whitegrid')
#     # sns.set_palette('deep', desat=.6)
#     sns.set_context("notebook", font_scale=1.5,
#                     rc={"lines.linewidth": 2.5})
#     Axes3D.get_proj = get_proj
#     x = np.array(x.tolist())  # x, y, t, p
#     mask = (x[:, 3] == 1)
#     x_pos = x[mask]
#     x_neg = x[mask == False]
#     pos_idx = np.random.choice(x_pos.shape[0], 10000)
#     neg_idx = np.random.choice(x_neg.shape[0], 10000)
#     # x_pos[pos_idx, 2] = 0
#     # x_neg[neg_idx, 2] = 0
#
#     fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
#     ax = Axes3D(fig)
#     ax.pbaspect = np.array([2.0, 1.0, 0.5])
#     ax.view_init(elev=10, azim=-75)
#     # ax.view_init(elev=15, azim=15)
#     ax.set_xlabel('t (time step)')
#     ax.set_ylabel('w (pixel)')
#     ax.set_zlabel('h (pixel)')
#     # ax.set_xticks([])
#     # ax.set_yticks([])
#     # ax.set_zticks([])
#     # ax.scatter(x_pos[pos_idx, 2], 48 - x_pos[pos_idx, 0], 48 - x_pos[pos_idx, 1], color='red', alpha=0.3, s=1.)
#     # ax.scatter(x_neg[neg_idx, 2], 48 - x_neg[neg_idx, 0], 48 - x_neg[neg_idx, 1], color='blue', alpha=0.3, s=1.)
#
#
#     ax.scatter(x_pos[:, 0], 120 - x_pos[:, 1] // 2, 80 - x_pos[:, 2], color='red', alpha=0.3, s=1.)
#     ax.scatter(x_neg[:, 0], 120 - x_neg[:, 1] // 2, 80 - x_neg[:, 2], color='blue', alpha=0.3, s=1.)
#     ax.scatter(18000, 120 - x_pos[:, 1] // 2, 80 - x_pos[:, 2], color='red', alpha=0.3, s=1.)
#     ax.scatter(18000, 120 - x_pos[:, 1] // 2, 80 - x_pos[:, 2], color='blue', alpha=0.3, s=1.)
#
#     # ax.scatter(x_pos[:, 0], 64 - x_pos[:, 1] // 2, 128 - x_pos[:, 2], color='red', alpha=0.3, s=1.)
#     # ax.scatter(x_neg[:, 0], 64 - x_neg[:, 1] // 2, 128 - x_neg[:, 2], color='blue', alpha=0.3, s=1.)
#     # ax.scatter(18000, 64 - x_pos[:, 1] // 2, 128 - x_pos[:, 2], color='red', alpha=0.3, s=1.)
#     # ax.scatter(18000, 64 - x_pos[:, 1] // 2, 128 - x_pos[:, 2], color='blue', alpha=0.3, s=1.)

def event_frame_plot_2d(event, s=''):
    if not isinstance(event, torch.BoolTensor):
        event = event.abs()

    # r = list(range(3)) + list(range(event.shape[0]-3, event.shape[0]))
    for t in range(event.shape[0]):
        pos_idx = []
        neg_idx = []
        for x in range(event.shape[2]):
            for y in range(event.shape[3]):
                if event[t, 0, x, y] > 0:
                    pos_idx.append((x, y, event[t, 0, x, y]))
                if event[t, 1, x, y] > 0:
                    neg_idx.append((x, y, event[t, 0, x, y]))
        if len(pos_idx) > 0:
            # print(t)
            pos_x, pos_y, pos_c = np.split(np.array(pos_idx), 3, axis=1)
            # print(pos_c)
            # print(pos_c.max())
            pos_c[pos_c > 3.] = 3.
            plt.figure(facecolor='black')
            plt.tick_params(bottom=False, top=False, left=False, right=False)
            plt.tick_params(bottom=False, top=False, left=False, right=False)
            plt.axis('off')
            plt.xlim(0, 48)
            plt.ylim(0, 48)
            if 'mask' in s:
                plt.scatter(pos_y[:, 0], 48 - pos_x[:, 0], c='oldlace', alpha=1, s=9)
            elif 'flatten' in s:
                neg_x, neg_y, neg_c = np.split(np.array(neg_idx), 3, axis=1)
                plt.scatter(pos_y[:, 0], 48 - pos_x[:, 0], c='red', alpha=1, s=1)
                plt.scatter(neg_y[:, 0], 48 - neg_x[:, 0], c='blue', alpha=1, s=1)
                plt.savefig('figure/flatten_%s_%d_%d.pdf' % (s, t, 0))
                plt.show()
                continue
            else:
                plt.scatter(pos_y[:, 0], 48 - pos_x[:, 0], c=pos_c, alpha=1, s=2, cmap='hot')
            plt.savefig('figure/%s_%d_%d_2d.pdf' % (s, t, 1), facecolor='black')
            plt.show()

        if len(neg_idx) > 0:
            neg_x, neg_y, neg_c = np.split(np.array(neg_idx), 3, axis=1)
            # print(neg_c.max())
            neg_c[neg_c > 3.] = 3.
            plt.figure(facecolor='black')
            plt.tick_params(bottom=False, top=False, left=False, right=False)
            plt.tick_params(bottom=False, top=False, left=False, right=False)
            plt.axis('off')
            plt.xlim(0, 48)
            plt.ylim(0, 48)
            if 'mask' in s:
                plt.scatter(neg_y[:, 0], 48 - neg_x[:, 0], c='oldlace', alpha=1, s=9)
            else:
                plt.scatter(neg_y[:, 0], 48 - neg_x[:, 0], c=neg_c, alpha=1, s=2, cmap='hot')
            plt.savefig('figure/%s_%d_%d_2d.pdf' % (s, t, 1), facecolor='black')
            plt.show()


dataset = get_dataloader_ncal(100, event_mix=False, beta=1., prob=1., num_classes=20, noise=0., num=1)
# dataset, _, _, _ = get_dvsc10_data()

# from tqdm import tqdm
# for i in tqdm(range(5000, 6000)):
#     img1, label = dataset[i]
#     ax = event_vis_raw_1d(img1)
#     plt.savefig('fig/temp_fig/dog_{}.jpg'.format(i-5000))
    # plt.show()
    # ax = event_vis_raw(img1)
    # plt.show()

# for batch_idx, (inputs, label) in enumerate(dataset):
#     # inputs = torch.sum(torch.sum(inputs[0], dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1)
#     inputs = torch.sum(torch.sum(inputs[0], dim=0), dim=0).unsqueeze(0).repeat(3, 1, 1)
#     inputs = cv2.resize(inputs.permute(1, 2, 0).numpy().astype('float'), (32, 32))
#     plt.figure()
#     plt.imshow(inputs)
#     plt.axis('off')
#     plt.show()
    # ax = event_vis_raw_1d(inputs)


import tqdm
tmp_idx = 6808
for i in tqdm.tqdm(range(tmp_idx, tmp_idx+1)):
    img1, label = dataset[i]
    plt.figure(figsize=(8, 6))
    # plt.ylim(bottom=0.)
    # plt.axis('off')
    ax = event_vis_raw_1d(img1)
    # plt.close()
    # plt.savefig('fig/gradcam_ncaltech101_origin/label_{}_id_{}.jpg'.format(label, i),
    #             bbox_inches='tight', pad_inches=0)
    # plt.close()
    # plt.show()
    ax = event_vis_raw(img1)
    # plt.show()
    # plt.savefig('fig/gradcam_ncaltech101_origin/{}.jpg'.format(i))
# print(max_x)
# print(max_y)
# plt.show()
# ax = event_vis_raw(img1)
# plt.show()
#
# img1, label = dataset[7801]
# ax = event_vis_raw_1d(img1)
# plt.show()
# ax = event_vis_raw(img1)
# plt.show()


# for idx, (img1, label) in enumerate(dataset):
#     # raw
#     print("idx:{}, label:{}".format(idx, label))
#     if idx >= 6000 and idx <= 8000:
#         ax = event_vis_raw_1d(img1)
#         plt.show()
#         ax = event_vis_raw(img1)