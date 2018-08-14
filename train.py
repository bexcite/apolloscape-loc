import argparse
from datetime import datetime

import numpy as np

import os
import time

import torch
from torch import nn
from torchvision import transforms, models
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# from torchviz import make_dot

from utils.training import train, validate, model_results_pred_gt

from datasets.apolloscape import Apolloscape

from utils.common import draw_poses
from utils.common import draw_record
from utils.common import imshow
from utils.common import save_checkpoint
# from utils.common import AverageMeter
from utils.common import calc_poses_params, quaternion_angular_error
from utils.common import draw_pred_gt_poses

from models.posenet import PoseNet, PoseNetCriterion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import warnings

def get_args():
    parser = argparse.ArgumentParser(description="Train localization network on Apolloscape dataset")
    parser.add_argument("--data", metavar="DIR", required=True,
                        help="Path to Apolloscape dataset")
    parser.add_argument("--road", metavar="ROAD_DIR", default="zpark-sample",
                        help="Path to the road within ApolloScape. Default: zpark-sample")
    parser.add_argument("--output-dir", metavar="OUTPUT_DIR", default="output_data",
                        help="Path to save logs, models, figures and videos. Default: output_data")
    parser.add_argument("--video", metavar="VIDEO_OUT_FILE", type=str, action="store", default=None, const="", nargs="?",
                        help="Generate and save video of a training")
    parser.add_argument("--no-display", dest="no_display", action="store_true", default=False,
                        help="Don't show any graphs")
    parser.add_argument("--no-cache-transform", dest="no_cache_transform", action="store_true", default=False,
                        help="Don't save cache of transformed images (saves lots of disk space but dramatically decreases the speed of training)")
    parser.add_argument("--device", metavar="DEVICE", default="cuda", type=str, choices=('cuda', 'cpu'),
                        help="Device to work on. Default: cuda")
    parser.add_argument("--checkpoint", metavar="CHECKPOINT_FILE", type=str,
                        help="Checkpoint file to restore model and optimizer parameters from")
    parser.add_argument("--checkpoint-save", metavar="EPOCH_NUM", type=int, default=100,
                        help="Save checkpoint every EPOCH_NUM epochs. Default: 100")
    parser.add_argument("--fig-save", metavar="EPOCH_NUM", type=int, default=0,
                        help="Save pred/gt figure on training dataset every EPOCH_NUM epochs. \
                            Default: 0 = don't save")
    parser.add_argument("--epochs", metavar="NUM_EPOCHS", type=int, default=1,
                        help="Number of epochs to train the model. Default: 1")
    parser.add_argument("--val-freq", metavar="VAL_FREQ", type=int, default=5,
                        help="Validation frequency every VAL_FREQ epochs. Default: 5")
    parser.add_argument("--log-freq", metavar="LOG_FREQ", type=int, default=0,
                        help="Log frequency during training and validation every LOG_FREQ batch. \
                        (default: 0 - once per epoch)")
    parser.add_argument("--batch-size", metavar="BATCH_SIZE", type=int, default=40,
                        help="Batch size. \
                        Default: 40 - fits in most cases on GPU")
    parser.add_argument("--lr", metavar="LR", type=float, default=1e-4,
                        help="Learning rate. \
                        Default: 1e-4")
    parser.add_argument("--beta", metavar="BETA", type=float, default=512.0,
                        help="Beta for geometric loss functions L = L(t) + beta * L(r) \
                        Default: 512.0")
    parser.add_argument("--learn-beta", dest="learn_beta", action="store_true", default=False,
                help="Automatically learn error weights for L(t) and L(r) instead of fixed beta. \
                Default: False")
    parser.add_argument("--experiment", metavar="EXP_NAME", type=str, default='run',
                        help="Experiment name. Defaul: run")
    parser.add_argument("--feature-net", metavar="FEATURE_NETWORK_NAME", default="resnet18",
                type=str, choices=('resnet18', 'resnet34', 'resnet50'),
                help="Feature extractor network. Choice from ('resnet18', 'resnet34', 'resnet50'). Default: resnet18")
    parser.add_argument("--feature-net-pretrained", dest="pretrained",
                action="store_true", default=False,
                help="Don't save cache of transformed images (saves lots of \
                disk space but dramatically decreases the speed of training)")
    parser.add_argument("--feature-net-features", metavar="NUM_FEATURES", type=int, default=2048,
                help="Number of features before the last regressor layer. Default: 2048")
    parser.add_argument("--stereo", dest="stereo", action="store_true", default=False,
                    help="Use stereo pairs for training (no geometric constraints applied). Default: False")






    return parser.parse_args()

def main():
    args = get_args()

    print('----- Params for debug: ----------------')
    print(args)

    print('data = {}'.format(args.data))
    print('road = {}'.format(args.road))

    print('Train model ...')

    # Imagenet normalization in case of pre-trained network
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Resize data before using
    transform = transforms.Compose([
        transforms.Resize(260),
        transforms.CenterCrop(250),
        transforms.ToTensor(),
        normalize
    ])


    train_record = None # 'Record001'
    train_dataset = Apolloscape(root=args.data, road=args.road,
        transform=transform, record=train_record, normalize_poses=True,
        pose_format='quat', train=True, cache_transform=not args.no_cache_transform,
        stereo=args.stereo)

    val_record = None # 'Record011'
    val_dataset = Apolloscape(root=args.data, road=args.road,
        transform=transform, record=val_record, normalize_poses=True,
        pose_format='quat', train=False, cache_transform=not args.no_cache_transform,
        stereo=args.stereo)


    # Show datasets
    print(train_dataset)
    print(val_dataset)

    shuffle_data = True

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_data) # batch_size = 75
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle_data) # batch_size = 75

    # Get mean and std from dataset
    poses_mean = val_dataset.poses_mean
    poses_std = val_dataset.poses_std

    # Select active device
    if torch.cuda.is_available() and args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device = {}'.format(device))


    # Used as prefix for filenames
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')


    # Create pretrained feature extractor
    if args.feature_net == 'resnet18':
        feature_extractor = models.resnet18(pretrained=args.pretrained)
    elif args.feature_net == 'resnet34':
        feature_extractor = models.resnet34(pretrained=args.pretrained)
    elif args.feature_net == 'resnet50':
        feature_extractor = models.resnet50(pretrained=args.pretrained)

    # Num features for the last layer before pose regressor
    num_features = args.feature_net_features # 2048

    experiment_name = get_experiment_name(args)

    # Create model
    model = PoseNet(feature_extractor, num_features=num_features)
    model = model.to(device)

    # Criterion
    criterion = PoseNetCriterion(stereo=args.stereo, beta=args.beta, learn_beta=args.learn_beta)
    criterion.to(device)
    
    # Add all params for optimization
    param_list = [{'params': model.parameters()}]
    if criterion.learn_beta:
        param_list.append({'params': criterion.parameters()})


    # Create optimizer
    optimizer = optim.Adam(params=param_list, lr=args.lr, weight_decay=0.0005)

    start_epoch = 0

    # Restore from checkpoint is present
    if args.checkpoint is not None:
        checkpoint_file = args.checkpoint

        if os.path.isfile(checkpoint_file):
            print('\nLoading from checkpoint: {}'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            start_epoch = checkpoint['epoch']
            if 'criterion_state_dict' in checkpoint:
                criterion.load_state_dict(checkpoint['criterion_state_dict'])
                print('Loaded criterion params too.')


    n_epochs = start_epoch + args.epochs

    print('\nTraining ...')
    val_freq = args.val_freq
    for e in range(start_epoch, n_epochs):

        # Train for one epoch
        train(train_dataloader, model, criterion, optimizer, e, n_epochs,
              log_freq=args.log_freq, poses_mean=train_dataset.poses_mean,
              poses_std=train_dataset.poses_std, device=device,
              stereo=args.stereo)

        # Run validation loop
        if e > 0 and e % val_freq == 0:
            end = time.time()
            validate(val_dataloader, model, criterion, e, log_freq=args.log_freq,
                    device=device, stereo=args.stereo)

        # Make figure
        if e > 0 and args.fig_save > 0 and e % args.fig_save == 0:
            exp_name = '{}_{}'.format(time_str, experiment_name)
            make_figure(model, train_dataloader, poses_mean=poses_mean,
                    poses_std=poses_std, epoch=e,
                    experiment_name=exp_name, device=device, stereo=args.stereo)

        # Make checkpoint
        if e > 0 and e % args.checkpoint_save == 0:
            make_checkpoint(model, optimizer, criterion, epoch=e, time_str=time_str,
                        args=args)


    print('\nn_epochs = {}'.format(n_epochs))



    print('\n=== Test Training Dataset ======')
    pred_poses, gt_poses = model_results_pred_gt(model, train_dataloader, poses_mean, poses_std,
                                                 device=device, stereo=args.stereo)

    print('gt_poses = {}'.format(gt_poses.shape))
    print('pred_poses = {}'.format(pred_poses.shape))
    t_loss = np.asarray([np.linalg.norm(p - t) for p, t in zip(pred_poses[:, :3], gt_poses[:, :3])])
    q_loss = np.asarray([quaternion_angular_error(p, t) for p, t in zip(pred_poses[:, 3:], gt_poses[:, 3:])])

    print('poses_std = {:.3f}'.format(np.linalg.norm(poses_std)))
    print('T: median = {:.3f}, mean = {:.3f}'.format(np.median(t_loss), np.mean(t_loss)))
    print('R: median = {:.3f}, mean = {:.3f}'.format(np.median(q_loss), np.mean(q_loss)))

    # Save for later visualization
    pred_poses_train = pred_poses
    gt_poses_train = gt_poses


    print('\n=== Test Validation Dataset ======')
    pred_poses, gt_poses = model_results_pred_gt(model, val_dataloader, poses_mean, poses_std,
                                                 device=device, stereo=args.stereo)

    print('gt_poses = {}'.format(gt_poses.shape))
    print('pred_poses = {}'.format(pred_poses.shape))
    t_loss = np.asarray([np.linalg.norm(p - t) for p, t in zip(pred_poses[:, :3], gt_poses[:, :3])])
    q_loss = np.asarray([quaternion_angular_error(p, t) for p, t in zip(pred_poses[:, 3:], gt_poses[:, 3:])])

    print('poses_std = {:.3f}'.format(np.linalg.norm(poses_std)))
    print('T: median = {:.3f}, mean = {:.3f}'.format(np.median(t_loss), np.mean(t_loss)))
    print('R: median = {:.3f}, mean = {:.3f}'.format(np.median(q_loss), np.mean(q_loss)))

    # Save for later visualization
    pred_poses_val = pred_poses
    gt_poses_val = gt_poses

    # Save checkpoint
    print('\nSaving model params ....')
    make_checkpoint(model, optimizer, criterion, epoch=n_epochs, time_str=time_str,
                    args=args)


def get_experiment_name(args):
    if args is not None:
        fname = '{}_{}'.format(args.experiment, args.feature_net)
        if args.pretrained:
            fname += 'p'
        fname += '_{}'.format(args.feature_net_features)
    else:
        fname = 'run'
    return fname


def make_checkpoint(model, optimizer, criterion, epoch=None, time_str=None, args=None):
    fname = get_experiment_name(args)
    saved_path = save_checkpoint(model, optimizer, criterion, experiment_name=fname,
                    epoch=epoch, time_str=time_str)

    print('Model saved to {}'.format(saved_path))


def make_figure(model, dataloader, poses_mean=None, poses_std=None,
        epoch=None, experiment_name=None, device='cpu', stereo=True):

    pred_poses, gt_poses = model_results_pred_gt(model, dataloader,
            poses_mean, poses_std, device=device, stereo=stereo)
    t_loss = np.asarray([np.linalg.norm(p - t) for p, t in zip(pred_poses[:, :3], gt_poses[:, :3])])
    q_loss = np.asarray([quaternion_angular_error(p, t) for p, t in zip(pred_poses[:, 3:], gt_poses[:, 3:])])

    draw_pred_gt_poses(pred_poses, gt_poses)
    plt.title('Prediction on Train: ep={}, Te={:.3f}, Re={:.3f}'.format(epoch, np.mean(t_loss), np.mean(q_loss)))

    if experiment_name:
        fig_dir = os.path.join('_checkpoints', experiment_name)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_path = os.path.join(fig_dir, '{}_e{}.png'.format(experiment_name, epoch))
        plt.savefig(fig_path)
        plt.close()
#         print("Fig saved to '{}'".format(fig_path))




if __name__ == "__main__":
    main()
