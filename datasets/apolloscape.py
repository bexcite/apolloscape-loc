import torch
import numpy as np
import transforms3d.euler as txe
import transforms3d.quaternions as txq
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.common import calc_poses_params, extract_translation
from PIL import Image
import os
import glob
import csv
import warnings
import pickle
import time


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


def read_all_data(image_dir, pose_dir, records_list, cameras_list,
        apollo_original_order=False, stereo=True):
    # iterate over all records and store it in internal data
#     data = []
    d_images = []
    d_poses = np.empty((0, 4, 4))
    d_records = []
    skipped_inc = 0
    skipped_other = 0
    for i, r in enumerate(records_list):
        cam1s = sorted(glob.glob(os.path.join(image_dir, r, cameras_list[0], '*.jpg')),
                       reverse=not apollo_original_order)
        cam2s = sorted(glob.glob(os.path.join(image_dir, r, cameras_list[1], '*.jpg')),
                       reverse=not apollo_original_order)

        # Read poses for first camera
        pose1s = read_poses_for_camera(os.path.join(pose_dir, r), cameras_list[0])

        # Read poses for second camera
        pose2s = read_poses_for_camera(os.path.join(pose_dir, r), cameras_list[1])

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

            if stereo and im1_part != im2_part:
                # Non-consistent images, drop with the lowest time unit
                # and repeat with the next idx
                skipped_inc += 1
                if im1_part < im2_part:
                    c1_idx += 1
                else:
                    c2_idx += 1
            else:

                # Images has equal timing (filename prefix) so add them to data.
                
                # First image
                d_images.append(c1)
                d_poses = np.vstack((d_poses, np.expand_dims(pose1s[os.path.basename(c1)], axis=0)))
                d_records.append(r)

                # Second image
                d_images.append(c2)
                d_poses = np.vstack((d_poses, np.expand_dims(pose2s[os.path.basename(c2)], axis=0)))
                d_records.append(r)


                # Continue with the next pair of images
                c1_idx += 1
                c2_idx += 1
    return np.array(d_images), d_poses, np.array(d_records)




def process_poses(all_poses, pose_format='full-mat',
                  normalize_poses=True):


    # pose_format value here is the default(current) representation
    _, _, poses_mean, poses_std = calc_poses_params(all_poses, pose_format='full-mat')

    # print('poses_mean = {}'.format(poses_mean))
    # print('poses_std = {}'.format(poses_std))


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
            # Constrain rotations to one hemisphere
            q *= np.sign(q[0])
            new_poses[i, :3] = t
            new_poses[i, 3:] = q
        all_poses = new_poses


    if normalize_poses:
#         print('Poses Normalized! pose_format = {}'.format(pose_format))
        if pose_format == 'quat':
            all_poses[:, :3] -= poses_mean
            all_poses[:, :3] = np.divide(all_poses[:, :3], poses_std, where=poses_std!=0)
        else: # 'full-mat'
            all_poses[:, :3, 3] -= poses_mean
            all_poses[:, :3, 3] = np.divide(all_poses[:, :3, 3], poses_std, where=poses_std!=0)

    # print('all_poses samples = {}'.format(all_poses[:10]))

    return all_poses, poses_mean, poses_std


def read_original_splits(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        return set(lines)


def get_rec_path(image_path):
    """Returns image path part {Record}/{Camera}/{Image}"""
    cp = os.path.splitext(image_path)[0]
    cp = os.path.normpath(cp).split(os.sep)
    cp = '/'.join(cp[-3:])
    return cp


def transforms_to_id(transforms):
    tname = ''
    for t in transforms.transforms:
        # print(t.__class__.__name__)
        if len(tname) > 0:
            tname += '_'
        tname += t.__class__.__name__
        if t.__class__.__name__ == 'Resize':
            tname += '_{}'.format(t.size)
        if t.__class__.__name__ == 'CenterCrop':
            tname += '_{}x{}'.format(t.size[0], t.size[1])
    return tname


class Apolloscape(Dataset):
    """Baidu Apolloscape dataset"""

    # Validatation ratio
    val_ratio = 0.25

    def __init__(self, root, road="road03_seg", transform=None, record=None,
                 normalize_poses=False, pose_format='full-mat', train=None,
                 cache_transform=False, stereo=True):
        """
            Args:
                root (string): Dataset directory
                road (string): Road subdir
                transform (callable, optional): A function/transform, similar to other PyTorch datasets
                record (string): Record name from dataset. Dataset organized in a structure '{road}/{recordXXX}'
                pose_format (string): One of 'full-mat', or 'quat'
                train (bool): default None - use all dataset, True - use just train dataset,
                              False - use val portion of a dataset if `train` is not None Records selection
                              are not applicable
                cache_transform (bool): Whether to save transformed images to disk. Helps to reduce \
                        computation needed during training by reusing already transformed and converted \
                        images from disk. (uses a lot of disk space, stores in '_cache_transform' folder
                stereo (bool): Retrun stereo pairs
        """
        self.root = os.path.expanduser(root)
        self.road = road
        self.transform = transform
        self.road_path = os.path.join(self.root, self.road)

        self.normalize_poses = normalize_poses
        self.pose_format = pose_format
        self.train = train
        self.apollo_original_order = True
        self.stereo = stereo

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

        self.metadata_road_dir = os.path.join('_metadata', road)


        # Read all data        
        self.d_images, self.d_poses, self.d_records = read_all_data(image_dir, pose_dir,
                self.records_list, self.cameras_list,
                apollo_original_order=self.apollo_original_order,
                stereo=self.stereo)
        

        # Process and convert poses
        self.d_poses, poses_mean, poses_std = process_poses(self.d_poses,
                pose_format=self.pose_format,
                normalize_poses=self.normalize_poses)
        self.poses_mean = poses_mean
        self.poses_std = poses_std
#         print('poses_mean = {}'.format(self.poses_mean))
#         print('poses_std = {}'.format(self.poses_std))


        # Store poses mean/std to metadata
        poses_stats_fname = 'pose_stats.txt'
        if not os.path.exists(self.metadata_road_dir):
            os.makedirs(self.metadata_road_dir)
        poses_stats_path = os.path.join(self.metadata_road_dir, poses_stats_fname)
        np.savetxt(poses_stats_path, (self.poses_mean, self.poses_std))


        # Check do we have train/val splits
        if self.train is not None:
            trainval_split_dir = os.path.join(self.road_path, "trainval_split")
            if not os.path.exists(trainval_split_dir):
                # Check do we have our own split
#                 print('check our own splits')
                trainval_split_dir = os.path.join(self.metadata_road_dir, "trainval_split")
                if not os.path.exists(trainval_split_dir):
                    # Create our own splits
#                     print('create our splits')
                    self.create_train_val_splits(trainval_split_dir)

            self.train_split = read_original_splits(os.path.join(trainval_split_dir, 'train.txt'))
            self.val_split = read_original_splits(os.path.join(trainval_split_dir, 'val.txt'))


        # Filter Train/Val
        if self.train is not None:
            def check_train_val(*args):
                result = True
                for a in args:
                    cp = get_rec_path(a)
                    result = result and self.check_test_val(cp)
                return result
            idxs = [i for i, r in enumerate(self.d_images) if check_train_val(r)]
            self.d_images = self.d_images[idxs]
            self.d_poses = self.d_poses[idxs]
            self.d_records = self.d_records[idxs]


        # Used as a filter in __len__ and __getitem__
        self.record = record


        # Cache transform directory prepare
        self.cache_transform = cache_transform
        if self.cache_transform:
            self.cache_transform_dir = \
                os.path.join('_cache_transform', self.road, transforms_to_id(self.transform))
#             print('cache_transform_dir = {}'.format(self.cache_transform_dir))


    def check_test_val(self, filename_path):
        """Checks whether to add image file to dataset based on Train/Val setting

        Args:
            filename_path (string): path in format ``{Record}/{Camera}/{image_name}.jpg``
        """
        if self.train is not None:
            # fname = os.path.splitext(filename_path)[0]
            fname = filename_path
            if self.train:
                return fname in self.train_split
            else:
                return fname in self.val_split
        else:
            return True


    def create_train_val_splits(self, trainval_split_dir):
        """Creates splits and saves it to ``train_val_split_dir``"""

        if not os.path.exists(trainval_split_dir):
            os.makedirs(trainval_split_dir)

        # Simply cut val_ratio for validation set
        l = int(len(self.d_images) * (1 - self.val_ratio))
        if l > 0 and l % 2 != 0:
            l = l - 1

        # Save train.txt
        with open(os.path.join(trainval_split_dir, 'train.txt'), 'w') as f:
            for s in self.d_images[:l]:
                f.write('{}\n'.format(get_rec_path(s)))
        # print('saved to {}'.format(os.path.join(trainval_split_dir, 'train.txt')))

        # Save val.txt
        with open(os.path.join(trainval_split_dir, 'val.txt'), 'w') as f:
            for s in self.d_images[l:]:
                f.write('{}\n'.format(get_rec_path(s)))
        # print('saved to {}'.format(os.path.join(trainval_split_dir, 'val.txt')))


    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, record):
        self.record_idxs = None
        self._record = record
        if self._record is not None:
            self.record_idxs = self.get_record_idxs(self._record)
            
            
    @property
    def d_poses_rec(self):
        if self.record_idxs is not None:
            return self.d_poses[self.record_idxs]
        return self.d_poses


    def poses_translations(self):
        """Get translation parts of the poses"""
        poses = [extract_translation(p, pose_format=self.pose_format) for p in self.d_poses_rec]
        return np.array(poses)


    def get_poses_params(self, all_records=False):
        """Returns min, max, mean and std values the poses translations"""
        poses = self.d_poses if all_records else self.d_poses_rec
        return calc_poses_params(poses, pose_format=self.pose_format)
    

    def get_records_counts(self):
        recs_num = {}
        for r in self.records_list:
            n = np.sum(self.d_records == r)
            if self.stereo:
                n = n // 2
            recs_num[r] = n
        return recs_num


    def get_record_idxs(self, record):
        """Returns idxs array for provided record."""
        if self.d_records is None:
            return None
        if record not in self.records_list:
            warnings.warn("Record '{}' does not exists in '{}'".format(
                self.record, os.path.join(self.root, self.road)))
        recs_filter = self.d_records == self.record
        all_idxs = np.arange(len(self.d_records))
        return all_idxs[recs_filter]


    @property
    def records(self):
        return self.records_list


    def __len__(self):
        l = len(self.d_images)
        if self.record_idxs is not None:
            l = len(self.record_idxs)
        if self.stereo:
            l = l // 2
        return l


    def load_image_direct(self, image_path):
        img = pil_loader(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img


    def load_image(self, image_path):

        if (self.transform is not None
                and self.cache_transform):

            # Using cache
            im_path_list = image_path.split(os.sep)
            cache_dir = os.path.join(self.cache_transform_dir, os.sep.join(im_path_list[-3:-1]))
            fname = im_path_list[-1] + '.pickle'
            cache_im_path = os.path.join(cache_dir, fname)
            if os.path.exists(cache_im_path):
                # return cached
#                 print('returned cached fname = {}'.format(cache_im_path))
                start_t = time.time()
                with open(cache_im_path, 'rb') as cache_file:
                    img = pickle.load(cache_file)
                return img

            # First time direct load
            start_t = time.time()
            img = self.load_image_direct(image_path)


            # Store to cache
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)

            with open(cache_im_path, 'wb') as cache_file:
                pickle.dump(img, cache_file, pickle.HIGHEST_PROTOCOL)

            return img

        # Not using cache at all
        start_t = time.time()
        img = self.load_image_direct(image_path)
        # print('T: load_direct = {:.3f}'.format(time.time() - start_t))

        return img

    # TODO: [TEST] Data 3 or 5 - support modes
    def __getitem__(self, idx):
        
        if self.stereo:
            idx = idx * 2
            
        ridx = idx

        # If we have a record than work with filtered self.record_idxs
        if self.record_idxs is not None:
            idx = self.record_idxs[ridx]
            
        img_path = self.d_images[idx]
        img = self.load_image(img_path)
        pos = torch.from_numpy(self.d_poses[idx])
        pos = pos.float()
            
        # Return one image (mono mode) 
        if not self.stereo:
            return img, pos
        
        # Second image (stereo mode)
        if self.record_idxs is not None:
            idx = self.record_idxs[ridx + 1]
        else:
            idx += 1
            
        img_path2 = self.d_images[idx]
        img2 = self.load_image(img_path2)
        pos2 = torch.from_numpy(self.d_poses[idx])
        pos2 = pos2.float()

        check_stereo_paths_consistency(img_path, img_path2)
        
        return [img, img2], [pos, pos2]


    def __repr__(self):
        fmt_str  = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Road: {}\n".format(self.road)
        fmt_str += "    Record: {}\n".format(self.record)
        fmt_str += "    Train: {}\n".format(self.train)
        fmt_str += "    Normalize Poses: {}\n".format(self.normalize_poses)
        fmt_str += "    Stereo: {}\n".format(self.stereo)
        fmt_str += "    Length: {} of {}\n".format(self.__len__(),
                            len(self.d_images) // 2 if self.stereo else len(self.d_images))
        fmt_str += "    Cameras: {}\n".format(self.cameras_list)
        fmt_str += "    Records: {}\n".format(self.records_list)
        return fmt_str
