import torch


class PoseNet(torch.nn.Module):

    def __init__(self, feature_extractor, num_features=128):
        super(PoseNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        fc_in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = torch.nn.Linear(fc_in_features, num_features)

        # Translation
        self.fc_xyz = torch.nn.Linear(num_features, 3)

        # Rotation in quaternions
        self.fc_quat = torch.nn.Linear(num_features, 4)

        # Turns off track_running_stats for BatchNorm layers,
        # it simplifies testing on small datasets due to eval()/train() differences
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = False

    def forward(self, x):
        # x is batch_images [batch_size x image, batch_size x image]

#         x = self.feature_extractor(x)

        x_features = [self.feature_extractor(xi) for xi in x]
        x_translations = [self.fc_xyz(xi) for xi in x_features]
        x_rotations = [self.fc_quat(xi) for xi in x_features]

        x_poses = [torch.cat((xt, xr), dim=1) for xt, xr in zip(x_translations, x_rotations)]

        return x_poses


class PoseNetCriterion(torch.nn.Module):
    def __init__(self, stereo=True, beta = 512):
        super(PoseNetCriterion, self).__init__()
        self.stereo = stereo
        self.loss_fn = torch.nn.L1Loss()
        self.beta = beta

    def forward(self, x, y):
        """
        Args:
            x: list(N x 7, N x 7) - prediction (xyz, quat)
            y: list(N x 7, N x 7) - target (xyz, quat)
        """

        loss = 0
        if self.stereo:
            for i in range(2):
                # Translation loss
                loss += self.loss_fn(x[i][:, :3], y[i][:, :3])
                # Rotation loss
#                 p_norm = torch.norm(x[i][:, 3:])
                p_norm = 1.0
                loss += self.beta * self.loss_fn(x[i][:, 3:]/p_norm, y[i][:, 3:])
        else:
            # Translation loss
            loss += self.loss_fn(x[:, :3], y[:, :3])

            # Rotation loss
            p_norm = torch.norm(x[:, 3:])
            loss += self.loss_fn(x[:, 3:]/p_norm, y[:, 3:])
#         print('x = \n{}'.format(x[0]))
#         print('y = \n{}'.format(y[0]))
        return loss
