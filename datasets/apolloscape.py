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
        apollo_original_order=False):
    # iterate over all records and store it in internal data
    data = []
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

            if im1_part != im2_part:
                # Non-consistent images, drop with the lowest time unit
                # and repeat with the next idx
                skipped_inc += 1
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
                data.append(item)

                # print('check1 = {}'.format(check1))
                # print(self.check_test_val(check1))
                # print('check2 = {}'.format(check2))
                # print(self.check_test_val(check2))

                # Continue with the next pair of images
                c1_idx += 1
                c2_idx += 1
    return data




def process_poses(all_poses, pose_format='full-mat',
                  normalize_poses=True):


#     print('process poses')


    # Mean/Std for full-mat
    # poses_mean = np.mean(all_poses[:, :3, 3], axis=0)
    # poses_std = np.std(all_poses[:, :3, 3], axis=0)

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
            new_poses[i, :3] = t
            new_poses[i, 3:] = q
#             if i == 0:
#                 print('R = {}'.format(R))
#                 print('t = {}'.format(t))
#                 print('q = {}'.format(q))
#                 print('t.shape = {}'.format(t.shape))
#                 print('new_poses[i] = {}'.format(new_poses[i]))
        all_poses = new_poses

        # poses_mean = np.mean(all_poses[:, :3], axis=0)
        # poses_std = np.std(all_poses[:, :3], axis=0)

        # _, _, poses_mean, poses_std = calc_poses_params(all_poses, pose_format=pose_format)
        # print('quat: poses_mean = {}'.format(poses_mean))
        # print('quat: poses_std = {}'.format(poses_std))


    if normalize_poses:
#         print('Poses Normalized! pose_format = {}'.format(pose_format))
        if pose_format == 'quat':
            # all_poses[:, :3] -= poses_mean[:3]
            # all_poses[:, :3] = np.divide(all_poses[:, :3], poses_std[:3], where=poses_std[:3]!=0)
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

    val_ratio = 0.25

    def __init__(self, root, road="road03_seg", transform=None, record=None,
                 normalize_poses=False, pose_format='full-mat', train=None, paired=True,
                 cache_transform=False):
        """
            Args:
                root (string): Dataset directory
                road (string): Road subdir
                transform (callable, optional): A function/transform, similar to other PyTorch datasets
                record (string): Record name from dataset. Dataset organized in a structure '{road}/{recordXXX}'
                pose_format (string): One of 'full-mat', 'rot-mat', 'quat', 'angles'
                train (bool): default None - use all dataset, True - use just train dataset,
                              False - use val portion of a dataset if `train` is not None Records selection
                              are not applicable
                paired (bool): Retrun stereo pairs
        """
        self.root = os.path.expanduser(root)
        self.road = road
        self.transform = transform
        self.road_path = os.path.join(self.root, self.road)

        self.normalize_poses = normalize_poses
        self.pose_format = pose_format
        self.train = train
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

        self.metadata_road_dir = os.path.join('_metadata', road)


        # Read all data
        self.data = read_all_data(image_dir, pose_dir, self.records_list, self.cameras_list,
                                  apollo_original_order=self.apollo_original_order)


        # Save for extracting poses directly
        self.data_array = np.array(self.data, dtype=object)


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
        
                
        # Store poses mean/std to metadata
        poses_stats_fname = 'pose_stats.txt'
        if not os.path.exists(self.metadata_road_dir):
            os.makedirs(self.metadata_road_dir)
        poses_stats_path = os.path.join(self.metadata_road_dir, poses_stats_fname)
        np.savetxt(poses_stats_path, (self.poses_mean, self.poses_std))


        # pp1, pp2 = self.poses_translations()
        # print('pp1 = {}'.format(pp1[:2]))
        # print('pp2 = {}'.format(pp2[:2]))


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
#                 else:
#                     print('we have our splits')
#             else:
#                 print('we have original splits')

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
            self.data = [r for r in self.data if check_train_val(r[0], r[2])]
            self._data_array = [r for r in self._data_array if check_train_val(r[0], r[2])]
            # Save for extracting poses directly
            self._data_array = np.array(self._data_array, dtype=object)


        # pp11, pp22 = self.poses_translations()
        # print('pp11 = {}'.format(pp11[:2]))
        # print('pp22 = {}'.format(pp22[:2]))

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

        # Simply cut val_ratio to validation
        l = int(len(self.data) * (1 - self.val_ratio))

        # Save train.txt
        with open(os.path.join(trainval_split_dir, 'train.txt'), 'w') as f:
            for s in self.data[:l]:
                f.write('{}\n'.format(get_rec_path(s[0])))
                f.write('{}\n'.format(get_rec_path(s[2])))
        # print('saved to {}'.format(os.path.join(trainval_split_dir, 'train.txt')))

        # Save val.txt
        with open(os.path.join(trainval_split_dir, 'val.txt'), 'w') as f:
            for s in self.data[l:]:
                f.write('{}\n'.format(get_rec_path(s[0])))
                f.write('{}\n'.format(get_rec_path(s[2])))
        # print('saved to {}'.format(os.path.join(trainval_split_dir, 'val.txt')))


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

        # print('=== poses translations = {}'.format(poses1[0]))

        # if self.pose_format == 'full-mat':
        poses1 = [extract_translation(p, pose_format=self.pose_format) for p in poses1]

        poses2 = [extract_translation(p, pose_format=self.pose_format) for p in poses2]



        return np.array(poses1), np.array(poses2)

    def all_poses(self):
        """Get all poses list for all records in dataset"""
        poses1 = self.all_data_array[:, 1]
        poses2 = self.all_data_array[:, 3]
        return poses1, poses2

    def get_poses_params(self, all_records=False):
        """Returns min, max, mean and std values the poses translations"""
        data_array = self.all_data_array if all_records else self.data_array
        poses1 = data_array[:, 1]
        poses2 = data_array[:, 3]
        all_poses = np.concatenate((poses1, poses2))
        return calc_poses_params(all_poses, pose_format=self.pose_format)


    def get_records_counts(self):
        data_array = self._data_array
        recs_num = {}
        for r in self.records_list:
            n = np.sum(data_array[:, 4] == r)
            recs_num[r] = n
        return recs_num



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


    def load_image_direct(self, image_path):
        img = pil_loader(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img


    def load_image(self, image_path):

        if (self.transform is not None
                and self.cache_transform):

            # Split to tensor
#             to_tensor = None
#             head_transform = self.transform
#             if head_transform.transforms[-1].__class__.__name__ == "ToTensor":
#                 to_tensor = head_transform.transforms[-1]
#                 head_transform = transforms.Compose(head_transform.transforms[:-1])


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
#                 img = pil_loader(image_path)
#                 print('T: cached hit pil_load = {:.3f}'.format(time.time() - start_t))
#                 if to_tensor is not None:
#                     img = to_tensor(img)
#                 print('T: cached hit = {:.3f}'.format(time.time() - start_t))
                return img

            # First time direct load
            start_t = time.time()
            img = self.load_image_direct(image_path)

#             img = pil_loader(image_path)
#             if head_transform is not None:
#                 img = head_transform(img)

            # Store to cache
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)

#             img.save(cache_im_path)

            with open(cache_im_path, 'wb') as cache_file:
                pickle.dump(img, cache_file, pickle.HIGHEST_PROTOCOL)

#             if to_tensor is not None:
#                 img = to_tensor(img)

#             print('T: load_direct + store = {:.3f}'.format(time.time() - start_t))

            return img

        # Not using cache at all
        start_t = time.time()
        img = self.load_image_direct(image_path)
        # print('T: load_direct = {:.3f}'.format(time.time() - start_t))

        return img


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
            img = self.load_image(ditem[im])
            images.append(img)
            npos = torch.from_numpy(ditem[pos])
            poses.append(npos.float())
        return images, poses


    def __repr__(self):
        fmt_str  = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    Road: {}\n".format(self.road)
        fmt_str += "    Record: {}\n".format(self.record)
        fmt_str += "    Train: {}\n".format(self.train)
        fmt_str += "    Normalize Poses: {}\n".format(self.normalize_poses)
        fmt_str += "    Length: {} of {}\n".format(self.__len__(), len(self.data))
        fmt_str += "    Cameras: {}\n".format(self.cameras_list)
        fmt_str += "    Records: {}\n".format(self.records_list)
        return fmt_str
