import torch
import numpy as np
import transforms3d.euler as txe
import transforms3d.quaternions as txq
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.common import calc_poses_params
from PIL import Image
import os
import glob
import csv
import warnings


def read_poses_dict(fname):
    """Reads poses.txt file from Apolloscape dataset
       Rotation in matrix 4x4 RT format.
    """
    poses = {}
    with open(fname) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            pose = np.asarray(row[:16], dtype=np.float).reshape(4, 4)
            poses[row[16]] = pose
    return poses


# Version for zpark sample type of structure
def read_poses_dict_6(fname):
    """Reads poses from alternative 'zpark-sample' format.
       Euler angles for rotations.
    """
    poses = {}
    with open(fname) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            pose = np.array(row[1].split(','), dtype=np.float)
            pose_rot = txe.euler2mat(*pose[0:3])
            pose_trans = pose[3:]
            full_pose = np.eye(4)
            full_pose[:3, :3] = pose_rot
            full_pose[:3, 3] = pose_trans
            poses[row[0]] = full_pose
    return poses


def read_poses_for_camera(record_path, camera_name):
    """Finds and reads poses file for camera_name."""

    # Resolve pose.txt file path for camera
    poses_path = os.path.join(record_path, camera_name, 'pose.txt')
    if os.path.exists(poses_path):
        poses = read_poses_dict(poses_path);
    else:
        # Sample type dataset (aka zpark-sample)
        poses_path = os.path.join(record_path, camera_name + '.txt')
        poses = read_poses_dict_6(poses_path);
    return poses



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def check_stereo_paths_consistency(c1, c2):
    """Check that left and right camera images has the same file name timestamp parts. """
    # TODO: May be differencesin 1-2 time units can be neglected?
    im1 = os.path.basename(c1).split('_')
    im2 = os.path.basename(c2).split('_')
    im1_part = '_'.join(im1[:2])
    im2_part = '_'.join(im2[:2])
    if im1_part != im2_part:
        warnings.warn("Not consistent stereo pair paths: \n{}\n{}".format(im1_part,
                                    im2_part))


def process_poses(all_poses, pose_format='full-mat',
                  normalize_poses=True):
    # Default pose format is full-mat
    new_poses = all_poses
    if pose_format == 'quat':
        # Convert to quaternions
        new_poses = np.zeros((len(all_poses), 7))
        for i in range(len(all_poses)):
            p = all_poses[i]
            R = p[:3, :3]
            t = p[:3, 3]
            q = txq.mat2quat(R)
            new_poses[i, :3] = t
            new_poses[i, 3:] = q
#             if i == 0:
#                 print('R = {}'.format(R))
#                 print('t = {}'.format(t))
#                 print('q = {}'.format(q))
#                 print('t.shape = {}'.format(t.shape))
#                 print('new_poses[i] = {}'.format(new_poses[i]))
        all_poses = new_poses

    poses_mean = np.mean(all_poses, axis=0)
    poses_std = np.std(all_poses, axis=0)

    if normalize_poses:
        print('Poses Normalized!')
        if pose_format == 'quat':
            all_poses[:, :3] -= poses_mean[:3]
            all_poses[:, :3] = np.divide(all_poses[:, :3], poses_std[:3], where=poses_std[:3]!=0)
        else: # 'full-mat'
            all_poses -= poses_mean
            all_poses = np.divide(all_poses, poses_std, where=poses_std!=0)

    return all_poses, poses_mean, poses_std



class Apolloscape(Dataset):
    """Baidu Apolloscape dataset"""

    def __init__(self, root, road="road03_seg", transform=None, record=None,
                 normalize_poses=False, pose_format='full-mat'):
        """
            Args:
                root (string): Dataset directory
                road (string): Road subdir
                transform (callable, optional): A function/transform, similar to other PyTorch datasets
                record (string): Record name from dataset. Dataset organized in a structure '{road}/{recordXXX}'
                pose_format (string): One of 'full-mat', 'rot-mat', 'quat', 'angles'
        """
        self.root = os.path.expanduser(root)
        self.road = road
        self.transform = transform
        self.road_path = os.path.join(self.root, self.road)

        self.normalize_poses = normalize_poses
        self.pose_format = pose_format
        self.apollo_original_order = True

        # Resolve image dir
        image_dir = os.path.join(self.road_path, "ColorImage")
        if not os.path.isdir(image_dir):
            # Sample type dataset (aka zpark-sample)
            image_dir = os.path.join(self.road_path, "image")

            # Reset flag, we will use it for Camera orders and images
            self.apollo_original_order = False
        if not os.path.isdir(image_dir):
            warnings.warn("Image directory can't be find in dataset path '{}'. " +
                          "Should be either 'ColorImage' or 'image'".format(self.road_path))


        # Resolve pose_dir
        pose_dir = os.path.join(self.road_path, "Pose")
        if not os.path.isdir(pose_dir):
            # Sample type dataset (aka zpark-sample)
            pose_dir = os.path.join(self.road_path, "pose")

            # Reset flag, we will use it for Camera orders and images
            self.apollo_original_order = False
        if not os.path.isdir(pose_dir):
            warnings.warn("Pose directory can't be find in dataset path '{}'. " +
                          "Should be either 'Pose' or 'pose'".format(self.road_path))

        self.records_list = [f for f in os.listdir(image_dir) if f not in [".DS_Store"]]
        self.records_list = sorted(self.records_list)


        if not len(self.records_list):
            warnings.warn("Empty records list in provided dataset '{}' for road '{}'".format(self.root, self.road))

        # NOTE: In zpark-sample cameras order is in right-left order,
        # which is an opposite of sorted as in original apollo datasets
        self.cameras_list = sorted(os.listdir(os.path.join(image_dir, self.records_list[0])),
                                   reverse=not self.apollo_original_order)

        # iterate over all records and store it in internal data
        self.data = []
        for i, r in enumerate(self.records_list):
            cam1s = sorted(glob.glob(os.path.join(image_dir, r, self.cameras_list[0], '*.jpg')),
                           reverse=not self.apollo_original_order)
            cam2s = sorted(glob.glob(os.path.join(image_dir, r, self.cameras_list[1], '*.jpg')),
                           reverse=not self.apollo_original_order)

            # Read poses for first camera
            pose1s = read_poses_for_camera(os.path.join(pose_dir, r), self.cameras_list[0])

            # Read poses for second camera
            pose2s = read_poses_for_camera(os.path.join(pose_dir, r), self.cameras_list[1])

            c1_idx = 0
            c2_idx = 0
            while c1_idx < len(cam1s) and c2_idx < len(cam2s):
                c1 = cam1s[c1_idx]
                c2 = cam2s[c2_idx]

                # Check stereo image path consistency
                im1 = os.path.basename(c1).split('_')
                im2 = os.path.basename(c2).split('_')
                im1_part = '_'.join(im1[:2])
                im2_part = '_'.join(im2[:2])

                if im1_part != im2_part:
                    # Non-consistent images, drop with the lowest time unit
                    # and repeat with the next idx
                    if im1_part < im2_part:
                        c1_idx += 1
                    else:
                        c2_idx += 1
                else:
                    # Images has equal timing (filename prefix) so add them to data.
                    item = []
                    item.append(c1)
                    item.append(pose1s[os.path.basename(c1)])
                    item.append(c2)
                    item.append(pose2s[os.path.basename(c2)])
                    item.append(r)
                    self.data.append(item)

                    # Continue with the next pair of images
                    c1_idx += 1
                    c2_idx += 1


        # Save for extracting poses directly
        self.data_array = np.array(self.data, dtype=object)

        # Used as a filter in __len__ and __getitem__
        self.record = record

        # Calc mean and std
        all_poses = np.empty((0, 4, 4))
        for p in np.concatenate((self._data_array[:,1], self._data_array[:,3])):
            all_poses = np.vstack((all_poses, np.expand_dims(p, axis=0)))

#         print('poses_poses = {}'.format(self.poses_mean))
#         print('poses_std = {}'.format(self.poses_std))

        # Process and convert poses
        all_poses_processed, poses_mean, poses_std = process_poses(all_poses,
                pose_format=self.pose_format,
                normalize_poses=self.normalize_poses)
        self.poses_mean = poses_mean
        self.poses_std = poses_std

        # Reassign poses after processing
        l = len(all_poses_processed)//2
        self._data_array[:,1] = [x for x in all_poses_processed[:l]]
        self._data_array[:,3] = [x for x in all_poses_processed[l:]]

#         if self.normalize_poses:
# #             print('Poses Normalized!')
#             all_poses -= self.poses_mean
#             all_poses = np.divide(all_poses, self.poses_std, where=self.poses_std!=0)
#             l = len(all_poses)//2
#             self._data_array[:,1] = [x for x in all_poses[:l]]
#             self._data_array[:,3] = [x for x in all_poses[l:]]

#         print('pose sample = {}'.format(self._data_array[0, 1]))
#         print('poses_mean = {}'.format(self.poses_mean))
#         print('poses_std = {}'.format(self.poses_std))


#         print('all_poses.len = {}'.format(len(all_poses)))
#         print('poses_poses = {}'.format(self.poses_mean))
#         print('poses_std = {}'.format(self.poses_std))

    @property
    def data_array(self):
        if hasattr(self, 'record_idxs') and self.record_idxs is not None:
            return self._data_array[self.record_idxs]
        return self._data_array

    @property
    def all_data_array(self):
        """Returns all data without record filters"""
        return self._data_array

    @data_array.setter
    def data_array(self, data_array):
        self._data_array = data_array


    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, record):
        self.record_idxs = None
        self._record = record
        if self._record is not None:
            self.record_idxs = self.get_record_idxs(self._record)


    def poses(self):
        """Get poses list filtered by current record"""
        poses1 = self.data_array[:, 1]
        poses2 = self.data_array[:, 3]
        return poses1, poses2

    def poses_translations(self):
        """Get translation parts of the poses"""
        poses1 = self.data_array[:, 1]
        poses2 = self.data_array[:, 3]
        if self.pose_format == 'full-mat':
            poses1 = [p[:3, 3] for p in poses1]
            poses2 = [p[:3, 3] for p in poses2]
            return np.array(poses1), np.array(poses2)
        # TODO: Implement for otherd pose_format - 'quat', etc

    def all_poses(self):
        """Get all poses list for all records in dataset"""
        poses1 = self.all_data_array[:, 1]
        poses2 = self.all_data_array[:, 3]
        return poses1, poses2

    def get_poses_params(self, all_records=False):
        """Returns min, max, mean and std values the poses"""
        data_array = self.all_data_array if all_records else self.data_array
        poses1 = data_array[:, 1]
        poses2 = data_array[:, 3]
        all_poses = np.concatenate((poses1, poses2))
        return calc_poses_params(all_poses)



    def get_record_idxs(self, record):
        """Returns idxs array for provided record."""
        if self.data_array is None:
            return None
        if record not in self.records_list:
            warnings.warn("Record '{}' does not exists in '{}'".format(
                self.record, os.path.join(self.root, self.road)))
        recs_filter = self.data_array[:, 4] == self.record
        all_idxs = np.arange(len(self.data_array))
        return all_idxs[recs_filter]


    @property
    def records(self):
        return self.records_list


    def __len__(self):
        if self.record_idxs is not None:
            return len(self.record_idxs)
        return len(self.data_array)


    def __getitem__(self, idx):

        # If we have a record than work with filtered self.record_idxs
        if self.record_idxs is not None:
            ditem = self._data_array[self.record_idxs[idx]]
        else:
            ditem = self._data_array[idx]

        # print("paths = \n{}\n{}".format(ditem[0], ditem[2]));

        check_stereo_paths_consistency(ditem[0], ditem[2])

        images = []
        poses = []
        for im, pos in zip([0,2], [1,3]):
            img = pil_loader(ditem[im])
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
            npos = torch.from_numpy(ditem[pos])
            poses.append(npos.float())
        return images, poses


    def __repr__(self):
        fmt_str  = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Road: {}\n".format(self.road)
        fmt_str += "    Record: {}\n".format(self.record)
        fmt_str += "    Length: {} of {}\n".format(self.__len__(), len(self.data))
        fmt_str += "    Normalize Poses: {}\n".format(self.normalize_poses)
        fmt_str += "    Cameras: {}\n".format(self.cameras_list)
        fmt_str += "    Records: {}\n".format(self.records_list)
        return fmt_str
