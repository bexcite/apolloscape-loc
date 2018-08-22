import torch
import torch.nn.functional as F


class PoseNet(torch.nn.Module):

    def __init__(self, feature_extractor, num_features=128, dropout=0.5,
                 track_running_stats=False, pretrained=False):
        super(PoseNet, self).__init__()
        self.dropout = dropout
        self.track_running_stats = track_running_stats
        self.pretrained = pretrained
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
                m.track_running_stats = self.track_running_stats
                
        # Initialization
        if self.pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_quat]
        else:
            init_modules = self.modules()
            
        for m in init_modules:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
                    
    def extract_features(self, x):
        x_features = self.feature_extractor(x)
        x_features = F.relu(x_features)
        if self.dropout > 0:
            x_features = F.dropout(x_features, p=self.dropout, training=self.training)
        return x_features

    def forward(self, x):
        # x is batch_images [batch_size x image, batch_size x image]

#         x = self.feature_extractor(x)

        if type(x) is list:
            x_features = [self.extract_features(xi) for xi in x]
            x_translations = [self.fc_xyz(xi) for xi in x_features]
            x_rotations = [self.fc_quat(xi) for xi in x_features]
            x_poses = [torch.cat((xt, xr), dim=1) for xt, xr in zip(x_translations, x_rotations)]  
        elif torch.is_tensor(x):
            x_features = self.extract_features(x)
            x_translations = self.fc_xyz(x_features) 
            x_rotations = self.fc_quat(x_features)
            x_poses = torch.cat((x_translations, x_rotations), dim=1)

        return x_poses


class PoseNetCriterion(torch.nn.Module):
    def __init__(self, stereo=True, beta = 512.0, learn_beta=False, sx=0.0, sq=-3.0):
        super(PoseNetCriterion, self).__init__()
        self.stereo = stereo
        self.loss_fn = torch.nn.L1Loss()
        self.learn_beta = learn_beta
        if not learn_beta:
            self.beta = beta
        else:
            self.beta = 1.0
        self.sx = torch.nn.Parameter(torch.Tensor([sx]), requires_grad=learn_beta)
        self.sq = torch.nn.Parameter(torch.Tensor([sq]), requires_grad=learn_beta)

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
                loss += torch.exp(-self.sx) * self.loss_fn(x[i][:, :3], y[i][:, :3]) + self.sx
                # Rotation loss
                loss += torch.exp(-self.sq) * self.beta * self.loss_fn(x[i][:, 3:], y[i][:, 3:]) + self.sq
        
            # Normalize per image so we can compare stereo vs no-stereo mode
            loss = loss / 2
        else:
            # Translation loss
            loss += torch.exp(-self.sx) * self.loss_fn(x[:, :3], y[:, :3])
            # Rotation loss
            loss += torch.exp(-self.sq) * self.beta * self.loss_fn(x[:, 3:], y[:, 3:]) + self.sq
#         print('x = \n{}'.format(x[0]))
#         print('y = \n{}'.format(y[0]))
        return loss
