from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *

import cv2
import math

#scaled image will be centrally cropped to this size
# max_h = 10000
# max_w = 10000

#ETH high res
max_h = 4096
max_w = 6144

base_image_size = 64 #cropped/scaled image size will be divisible by this number

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.0, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        self.scaling = 1.0
        self.augment_data = False
        self.depth_scaling = 1.0

        for key, value in kwargs.items(): 
            print ("%s == %s" %(key, value))
            if(key == 'scaling'):
                self.scaling = value
            if(key == 'augment_data'):
                self.augment_data = value
            if(key == 'depth_scaling'):
                self.depth_scaling = 1.0/value

        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        # print("filename: ", filename)
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])

        depth_interval = float(lines[11].split()[1])
        # print("depth_interval and scaling: ", depth_interval, self.interval_scale)
        depth_interval *= self.interval_scale
        # print("depth_interval after scaling: ", depth_interval)

        depth_num = float(lines[11].split()[2])
        depth_max = float(lines[11].split()[3])

        # print("depths: ", depth_min, depth_interval, depth_num, depth_max)

        return intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        # assert np_img.shape[:2] == (1200, 1600)

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_and_crop(self, scaling, img, intrinsics, depth_image=None):
        if(scaling != 1.0):
            img = cv2.resize(img, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
            intrinsics[:2, :] *= scaling
            if not depth_image is None:
                depth_image = cv2.resize(depth_image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)

        # crop images and cameras
        h, w = img.shape[0:2]
        new_h = h
        new_w = w
        # print("scale and crop h, w: ", h, w)
        if new_h > max_h:
            new_h = max_h
        else:
            new_h = int(math.floor(h / base_image_size) * base_image_size)
        if new_w > max_w:
            new_w = max_w
        else:
            new_w = int(math.floor(w / base_image_size) * base_image_size)
        # print("new h, w: ", new_h, new_w)
        start_h = int(math.ceil((h - new_h) / 2))
        start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        img = img[start_h:finish_h, start_w:finish_w]

        #intrinsics are already scaled by factor 4 ?!
        intrinsics[0,2] = intrinsics[0,2] - start_w 
        intrinsics[1,2] = intrinsics[1,2] - start_h

        # crop depth image
        if not depth_image is None:
            depth_image = depth_image[start_h:finish_h, start_w:finish_w]
            return img, intrinsics, depth_image
        else:
            return img, intrinsics

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max = self.read_cam_file(proj_mat_filename)
            img = self.read_img(img_filename)

            # print("img shape: ", img.shape)
            # print("intrinsics: ", intrinsics)
            # depth_max /= 4

            depth_interval_scaling = depth_num/self.ndepths

            # print("interval: ", depth_interval_scaling, depth_num, self.ndepths)

            depth_interval *= depth_interval_scaling

            # print("interval scaled: ", depth_interval)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)
                img, intrinsics = self.scale_and_crop(self.scaling, img, intrinsics, depth)
                print("image: ", img_filename, img.shape)
                # print("depth: ", depth_filename, depth.shape)
                # cv2.imshow("img", img)
                # cv2.imshow("depth", depth)
                # cv2.waitKey(0)
            else:
                img, intrinsics = self.scale_and_crop(self.scaling, img, intrinsics)
            
            imgs.append(img)

            # intrinsics and extrinsics separately
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        #ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 8
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 16
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 32
        stage4_pjmats = proj_matrices.copy()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 64

        stage5_pjmats = proj_matrices.copy()
        stage6_pjmats = proj_matrices.copy()
        stage6_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2

        proj_matrices_ms = {
            "stage0": stage0_pjmats,
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats,
            "stage4": stage4_pjmats,
            "stage5": stage5_pjmats,
            "stage6": stage6_pjmats
        }

        print("depth values: ", depth_values.shape)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
