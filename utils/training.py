from utils.common import AverageMeter
import numpy as np
import time

import torch

from utils.common import quaternion_angular_error


# train function
def train(train_loader, model, criterion, optimizer, epoch, max_epoch, log_freq=1, print_sum=True,
          poses_mean=None, poses_std=None, device=None):

    # switch model to training
    model.train()

    losses = AverageMeter()

    epoch_time = time.time()

    gt_poses = np.empty((0, 7))
    pred_poses = np.empty((0, 7))


    end = time.time()
    for idx, (batch_images, batch_poses) in enumerate(train_loader):
        data_time = (time.time() - end)

        # TODO: Stereo=False mode (make it Tensor???? instead of list)
        batch_images = [x.to(device) for x in batch_images]
        batch_poses = [x.to(device) for x in batch_poses]

        out = model(batch_images)
        loss = criterion(out, batch_poses)
#         print('loss = {}'.format(loss))

        # TODO: Stereo=False mode
        losses.update(loss, len(batch_images) * batch_images[0].size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # TODO: Stereo=False
        # move data to cpu & numpy
        bp = [x.detach().cpu().numpy() for x in batch_poses]
        outp = [x.detach().cpu().numpy() for x in out]
        gt_poses = np.vstack((gt_poses, *bp))
        pred_poses = np.vstack((pred_poses, *outp))


        batch_time = (time.time() - end)
        end = time.time()

        if log_freq != 0 and idx % log_freq == 0:
            print('Epoch: [{}/{}]\tBatch: [{}/{}]\t'
                  'Time: {batch_time:.3f}\t'
                  'Data Time: {data_time:.3f}\t'
                  'Loss: {losses.val:.3f}\t'
                  'Avg Loss: {losses.avg:.3f}\t'.format(
                   epoch, max_epoch - 1, idx, len(train_loader) - 1,
                   batch_time=batch_time, data_time=data_time, losses=losses))

        if idx == 0:
            break


    # un-normalize translation
    unnorm = (poses_mean is not None) and (poses_std is not None)
    if unnorm:
        gt_poses[:, :3] = gt_poses[:, :3] * poses_std + poses_mean
        pred_poses[:, :3] = pred_poses[:, :3] * poses_std + poses_mean

    t_loss = np.asarray([np.linalg.norm(p - t) for p, t in zip(pred_poses[:, :3], gt_poses[:, :3])])
    q_loss = np.asarray([quaternion_angular_error(p, t) for p, t in zip(pred_poses[:, 3:], gt_poses[:, 3:])])

#     if unnorm:
#         print('poses_std = {:.3f}'.format(np.linalg.norm(poses_std)))
#     print('T: median = {:.3f}, mean = {:.3f}'.format(np.median(t_loss), np.mean(t_loss)))
#     print('R: median = {:.3f}, mean = {:.3f}'.format(np.median(q_loss), np.mean(q_loss)))


    if print_sum:
        print('Ep: [{}/{}]\tTrain Loss: {:.3f}\tTe: {:.3f}\tRe: {:.3f}\t Et: {:.2f}s'.format(
            epoch, max_epoch - 1, losses.avg, np.mean(t_loss), np.mean(q_loss),
            (time.time() - epoch_time)))

#     return losses.avg


def validate(val_loader, model, criterion, epoch, log_freq=1, print_sum=True, device=None):

    losses = AverageMeter()

    # set model to evaluation
    model.eval()

    with torch.no_grad():
        epoch_time = time.time()
        end = time.time()
        for idx, (batch_images, batch_poses) in enumerate(val_loader):
            data_time = time.time() - end

            # TODO: Stereo=False mode support
            batch_images = [x.to(device) for x in batch_images]
            batch_poses = [x.to(device) for x in batch_poses]

            # compute model output
            out = model(batch_images)
            loss = criterion(out, batch_poses)

            # TODO: Stereo=False mode support
            losses.update(loss, len(batch_images) * batch_images[0].size(0))

            batch_time = time.time() - end
            end = time.time()

            if log_freq != 0 and idx % log_freq == 0:
                print('Val Epoch: {}\t'
                      'Time: {batch_time:.3f}\t'
                      'Data Time: {data_time:.3f}\t'
                      'Loss: {losses.val:.3f}\t'
                      'Avg Loss: {losses.avg:.3f}'.format(
                       epoch, batch_time=batch_time, data_time=data_time, losses=losses))

            if idx == 0:
                break


    if print_sum:
        print('Epoch: [{}]\tValidation Loss: {:.3f}\tEpoch time: {:.3f}'.format(epoch, losses.avg,
                                                                               (time.time() - epoch_time)))

#     return losses.avg


def model_results_pred_gt(model, dataloader, poses_mean=None, poses_std=None, device=None):
    model.eval()

    gt_poses = np.empty((0, 7))
    pred_poses = np.empty((0, 7))

    for idx, (batch_images, batch_poses) in enumerate(dataloader):

        # TODO: Stereo=False mode support
        batch_images = [x.to(device) for x in batch_images]
        batch_poses = [x.to(device) for x in batch_poses]

        out = model(batch_images)

        # loss = criterion(out, batch_poses)
#         print('loss = {}'.format(loss))

        # TODO: Stereo=False mode support
        # move data to cpu & numpy
        batch_poses = [x.detach().cpu().numpy() for x in batch_poses]
        out = [x.detach().cpu().numpy() for x in out]

        gt_poses = np.vstack((gt_poses, *batch_poses))
        pred_poses = np.vstack((pred_poses, *out))

        if idx == 0:
            break

    # un-normalize translation
    unnorm = (poses_mean is not None) and (poses_std is not None)
    if unnorm:
        gt_poses[:, :3] = gt_poses[:, :3] * poses_std + poses_mean
        pred_poses[:, :3] = pred_poses[:, :3] * poses_std + poses_mean

    return pred_poses, gt_poses
