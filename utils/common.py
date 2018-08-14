import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
import torch
from tqdm import tqdm
from PIL import ImageFile
import os
from datetime import datetime
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True # it's important because not all images read without this

# Normalization params for pretrained models
img_norm_mean=np.array([0.485, 0.456, 0.406])
img_norm_std=np.array([0.229, 0.224, 0.225])

def img_tensor_to_numpy(img, img_normalized=True):
    if type(img) == torch.Tensor:
        if img_normalized:
            mean_t = torch.FloatTensor(img_norm_mean).view(3, 1, 1)
            std_t = torch.FloatTensor(img_norm_std).view(3, 1, 1)
            img = img * std_t + mean_t
        img = img.cpu().numpy().transpose([1, 2, 0])
    return img

# Show images
def imshow(img, title=None, img_normalized=True):
#     img = img.numpy().transpose([1, 2, 0])
    img = img_tensor_to_numpy(img, img_normalized=img_normalized)
    fig = plt.figure(figsize=(18, 18))
    if title is not None:
        plt.title(title)
    plt.imshow(img)
    plt.pause(0.001)


def make_video(dataset, record=None, outfile=None):
    if record is not None:
        dataset.record = record

    if outfile is None:
        outfile = "./output_data/videos/{}_{}.mp4".format(dataset.road, dataset.record)


    # Make dirs for video if needed
    video_path = os.path.dirname(outfile)
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie {}, {}'.format(dataset.road, dataset.record),
                    artist='Apolloscape & Pavlo',
                    comment='Preview records')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure(figsize=(8, 8))

    if dataset.stereo:
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
    else:
        ax1 = plt.subplot(2, 1, 1)
    
    ax3 = plt.subplot(2, 1, 2, projection='3d')
    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    print("Saving video to: {}".format(outfile))

    # Video Loop
    with writer.saving(fig, outfile, 100):
        
        step = 1
        
        # Iterate over one camera only
        if not dataset.stereo:
            step = 2

        for idx in tqdm(range(0, len(dataset), step)):

            if dataset.stereo:
                draw_record(dataset, idx=idx, axes=[ax1, ax2, ax3])
            else:
                draw_record(dataset, idx=idx, axes=[ax1, ax3])

            # Store video frame
            writer.grab_frame()

    # Clear figure
    plt.close(fig)

    print("Video saved successfully!")


def draw_record(dataset, record=None, idx=None, restore_record=True, axes=None, img_normalized=True):
    
    # Save current dataset's record and restore it later
    if record is not None and restore_record:
        saved_rec = dataset.record
        dataset.record = record
        
    if len(dataset) == 0:
#         print('Empty dataset for record {}'.format(dataset.record))
        if restore_record:
            dataset.record = saved_rec
        return
        

    fig = None
    if axes is None:
        fig = plt.figure(figsize=(8, 8))
        if dataset.stereo:
            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
        else:
            ax1 = plt.subplot(2, 1, 1)
        ax3 = plt.subplot(2, 1, 2, projection='3d')
        plt.subplots_adjust(wspace=0.01, hspace=0.0)
    else:
        if dataset.stereo:
            ax1 = axes[0]
            ax2 = axes[1]
            ax3 = axes[2]
        else:
            ax1 = axes[0]
            ax3 = axes[1]

        # Clean previous data
        ax1.cla()
        if dataset.stereo:
            ax2.cla()
        ax3.cla()


    # Set up axes
    ax1.axis('off')
    if dataset.stereo:
        ax2.axis('off')
    ax3.set_xlabel('$X$')
    ax3.set_ylabel('$Y$')
    ax3.set_zlabel('$Z$')
    ax3.view_init(50, 30)


    # Sample a data point from the record
    if idx is None:
        idx = np.random.randint(len(dataset))
    images, poses = dataset[idx]


    p_min, p_max, p_mean, p_std = dataset.get_poses_params()
    # print("min = {}".format(p_min))
    # print("max = {}".format(p_max))
    # print("mean = {}".format(p_mean))
    # print("std = {}".format(p_std))

    ax3.set_title("{} {} ({} of {})".format(dataset.road,
        dataset.record if dataset.record is not None else "",
        idx, len(dataset)))

    # Set plot limits acc to selected poses
    ax3.set_xlim(int(p_min[0] - 1), int(p_max[0] + 1))
    ax3.set_ylim(int(p_min[1] - 1), int(p_max[1] + 1))
    ax3.set_zlim(int(p_min[2] - 1), int(p_max[2] + 1))

    # Show all poses for selected record
    all_poses = dataset.poses_translations()

    draw_poses(ax3, all_poses, proj=True, proj_z=int(p_min[2] - 1))

    # Show current sample pose
    if dataset.stereo:
        mid_pose = 0.5 * (extract_translation(poses[0], pose_format=dataset.pose_format)
                    + extract_translation(poses[1], pose_format=dataset.pose_format))
    else:
        mid_pose = extract_translation(poses, pose_format=dataset.pose_format)
    
    draw_poses(ax3, [mid_pose], c='r', s=60, proj=True, proj_z=int(p_min[2] - 1 ))

#     ax1.imshow(images[0])
#     ax2.imshow(images[1])
    if dataset.stereo:
        ax1.imshow(img_tensor_to_numpy(images[0], img_normalized=img_normalized))
        ax2.imshow(img_tensor_to_numpy(images[1], img_normalized=img_normalized))
    else:
        ax1.imshow(img_tensor_to_numpy(images, img_normalized=img_normalized))


    # Restore record if it was custom
    if record is not None and restore_record:
        dataset.record = saved_rec

    return fig


def draw_poses(ax, poses, c='b', s=20, proj=False, proj_z=0, pose_format='quat'):
    """Draws the list of poses.

    Args:
        ax (Axes3D): 3D axes
        poses (list): Poses list
        c: matplotlib color
        s: matplotlib size
        proj (bool): True if draw projection of a path on z-axis
        proj_z (float): Coord for z-projection
    """
    coords = np.zeros((len(poses), 3))
    for i, p in enumerate(poses):
        # coords[i] = p[:3, 3]
        # coords[i] = p
        coords[i] = extract_translation(p, pose_format=pose_format)

    # Draw projection
    if proj:
        if len(poses) > 1:
            ax.plot(coords[:, 0], coords[:, 1], proj_z, c='g')
        elif len(poses) == 1:
            ax.scatter(coords[:, 0], coords[:, 1], proj_z, c=c)

    # Draw path
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=c, s=s)


def draw_poses_list(ax, poses_list):
    """Draw list of lists of poses. Use to draw several paths."""
    for poses in poses_list:
        draw_poses(ax, poses)


def set_3d_axes_limits(ax, poses, pose_format='quat'):
    p_min, p_max, p_mean, p_std = calc_poses_params(poses, pose_format=pose_format)
    ax.set_xlim(p_min[0], p_max[0])
    ax.set_ylim(p_min[1], p_max[1])
    ax.set_zlim(int(p_min[2] - 1), p_max[2])
    return p_min, p_max, p_mean, p_std

def draw_pred_gt_poses(pred_poses, gt_poses):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.view_init(50, 30)

    all_poses = np.concatenate((pred_poses, gt_poses))
    p_min, _, _, _ = set_3d_axes_limits(ax, all_poses, pose_format='quat')

    draw_poses(ax, pred_poses[:, :3], proj=False, proj_z=int(p_min[2] - 1), c='r', s=60)
    draw_poses(ax, gt_poses[:, :3], proj=False, proj_z=int(p_min[2] - 1), c='b', s=60)
    for i in range(pred_poses.shape[0]):
        pp = pred_poses[i, :3]
        gp = gt_poses[i, :3]
        pps = np.vstack((pp, gp))
        ax.plot(pps[:, 0], pps[:, 1], pps[:, 2], c=(0.7, 0.7, 0.7, 0.4))

    plt.draw()



def extract_translation(p, pose_format='full-mat'):
    if pose_format == 'full-mat':
        return p[0:3, 3]
    elif pose_format == 'quat':
        return p[:3]
    else:
        warnings.warn("pose_format should be either 'full-mat' or 'quat'")
        return p


def calc_poses_params(poses, pose_format='full-mat'):
    """Calculates min, max, mean and std of translations of the poses"""

    # TODO: Make normal loop and use empty
    p = poses[0]
    allp = extract_translation(p, pose_format)

    for p in poses[1:]:
        allp = np.vstack((allp, extract_translation(p, pose_format)))

    p_min = np.min(allp, axis=0)
    p_max = np.max(allp, axis=0)
    p_mean = np.mean(allp, axis=0)
    p_std = np.std(allp, axis=0)

    return p_min, p_max, p_mean, p_std


# Save checkpoint
def save_checkpoint(model, optimizer, criterion, experiment_name='test', epoch=None,
                    time_str=None):
    if not time_str:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = '{}_{}'.format(time_str, experiment_name)
    if epoch is not None:
        fname += '_e{:03d}'.format(epoch)
    fname += '.pth.tar'

    checkpoints_dir = '_checkpoints'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    fname_path = os.path.join(checkpoints_dir, fname)

    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }
    
    if criterion.learn_beta:
        checkpoint_dict.update({'criterion_state_dict': criterion.state_dict()})

    torch.save(checkpoint_dict, fname_path)

    return fname_path




def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    abs_q1 = np.linalg.norm(q1)
    abs_q2 = np.linalg.norm(q2)
    d = d / (abs_q1 * abs_q2)
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.val = value
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count
