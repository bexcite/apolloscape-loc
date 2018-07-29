import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
import torch
from tqdm import tqdm
from PIL import ImageFile
import os
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True # it's important because not all images read without this

# Show images
def imshow(img, title=None):
    img = img.numpy().transpose([1, 2, 0])
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

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 1, 2, projection='3d')
    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    print("Saving video to: {}".format(outfile))

    # Video Loop
    with writer.saving(fig, outfile, 100):

        for idx in tqdm(range(len(dataset))):

            draw_record(dataset, idx=idx, axes=[ax1, ax2, ax3])

            # Store video frame
            writer.grab_frame()

    # Clear figure
    plt.close(fig)

    print("Video saved successfully!")


def draw_record(dataset, record=None, idx=None, restore_record=True, axes=None):
    # Save current dataset's record and restore it later
    if record is not None and restore_record:
        saved_rec = dataset.record
        dataset.record = record

    fig = None
    if axes is None:
        fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 1, 2, projection='3d')
        plt.subplots_adjust(wspace=0.01, hspace=0.0)
    else:
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        # Clean previous data
        ax1.cla()
        ax2.cla()
        ax3.cla()


    # Set up axes
    ax1.axis('off')
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
    poses1, poses2 = dataset.poses_translations()
    mid_poses = 0.5 * (poses1 + poses2)

    # print('mid_poses = {}'.format(mid_poses))

    draw_poses(ax3, mid_poses, proj=True, proj_z=int(p_min[2] - 1))

    # Show current sample pose
    mid_pose = 0.5 * (extract_translation(poses[0], pose_format=dataset.pose_format)
                + extract_translation(poses[1], pose_format=dataset.pose_format))
    draw_poses(ax3, [mid_pose], c='r', s=60, proj=True, proj_z=int(p_min[2] - 1 ))

    # Show current sample camera images
    if type(images[0]) == torch.Tensor:
        images[0] = images[0].numpy().transpose([1, 2, 0])
    if type(images[1]) == torch.Tensor:
        images[1] = images[1].numpy().transpose([1, 2, 0])

    ax1.imshow(images[0])
    ax2.imshow(images[1])


    # Restore record if it was custom
    if record is not None and restore_record:
        dataset.record = saved_rec

    return fig


def draw_poses(ax, poses, c='b', s=20, proj=False, proj_z=0):
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
        coords[i] = p

    # Draw projection
    if proj:
        if len(poses) > 1:
            ax.plot(coords[:, 0], coords[:, 1], proj_z, c='g')
        elif len(poses) == 1:
            ax.scatter(coords[:, 0], coords[:, 1], proj_z, c=c)

    # Draw path
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=c, s=s)

#     XYZ = np.tile(np.array([0., 0., 0., 1.]).reshape(4, -1), 1)
#         XYZp = np.matmul(p, XYZ)
#         ax.scatter(XYZp[0], XYZp[1], XYZp[2], c=c, s=s)


def draw_poses_list(ax, poses_list):
    """Draw list of lists of poses. Use to draw several paths."""
    for poses in poses_list:
        draw_poses(ax, poses)


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
def save_checkpoint(model, optimizer, experiment_name='test', epoch=None):
    tstr = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = '{}_{}'.format(tstr, experiment_name)
    if epoch is not None:
        fname += '_e{:03d}'.format(epoch)
    fname += '.pth.tar'

    checkpoints_dir = '_checkpoints'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    fname_path = os.path.join(checkpoints_dir, fname)
#     print('fname_path = {}'.format(fname_path))

    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }

    torch.save(checkpoint_dict, fname_path)

    return fname_path

    print('Model saved to {}'.format(fname_path))


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
