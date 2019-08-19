import torch

from models.base_model import BaseModel

class C3D(BaseModel):
    """
    The C3D network as described in [1].
    """

    def __init__(self, dropout_prob=0.5, pretrained=True, classifier=True, tpad=True):
        super(C3D, self).__init__()
        tpad = 2 if tpad else 1

        self.classifier = classifier

        self.conv1 = torch.nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = torch.nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = torch.nn.MaxPool3d(kernel_size=(tpad, 2, 2), stride=(2, 2, 2))

        self.conv3a = torch.nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = torch.nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = torch.nn.MaxPool3d(kernel_size=(tpad, 2, 2), stride=(2, 2, 2))

        self.conv4a = torch.nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = torch.nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = torch.nn.MaxPool3d(kernel_size=(tpad, 2, 2), stride=(2, 2, 2))

        self.conv5a = torch.nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = torch.nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = torch.nn.MaxPool3d(kernel_size=(tpad, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.relu = torch.nn.ReLU()

        if self.classifier:
            self.fc6 = torch.nn.Linear(8192, 4096)
            self.fc7 = torch.nn.Linear(4096, 4096)
            self.fc8 = torch.nn.Linear(4096, 487)

            self.dropout = torch.nn.Dropout(p=dropout_prob)

            # self.softmax = torch.nn.Softmax()

        if pretrained:
            self.load_weights('./weigths/c3d.pickle')

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        if self.classifier:
            h = h.view(-1, 8192)
            h = self.relu(self.fc6(h))
            h = self.dropout(h)
            h = self.relu(self.fc7(h))
            h = self.dropout(h)

            logits = self.fc8(h)

            return logits
        else:
            return h


class C3dNet(BaseModel):
    def __init__(self, n_labels, dropout_prob, pretrained=True, *args, **kwargs):
        super(C3dNet, self).__init__(*args, **kwargs)

        self.c3d = C3D(dropout_prob=dropout_prob, pretrained=pretrained)
        self.c3d.fc8 = torch.nn.Linear(4096, n_labels)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        y_pred = self.c3d(x)

        return y_pred

    def loss(self, y_preds, y):
        loss = self.criterion(y_preds, y)

        return loss
