import random
import numpy as np
import os
import cv2
import pandas as pd
import pprint
from functools import partial
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torchvision.models as models
import torchvision.transforms as transforms
import torchmetrics
import torchaudio


torchaudio.set_audio_backend("soundfile")


warnings.filterwarnings("ignore", message="No audio backend is available.")


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


args = {"data_folder": "D:\\abnormal_detection_dataset\\UBI_FIGHTS\\videos\\All\\",
        "graphs_folder": "./graph/4x3/resnet50/", "epoch": 1, "batch_size": 1, "num_class": 1, "learning_rate": 1e-4,
        "decay_rate": 0.0001, "num_workers": 4, "img_size": (640, 360), "img_depth": 3, "FPS": 10, "SEED": 42}

if not os.path.exists(args["graphs_folder"]): os.mkdir(args["graphs_folder"])

model_save_folder = './trained_model/4x3/'
if not os.path.exists(model_save_folder): os.mkdir(model_save_folder)
if not os.path.exists(args["graphs_folder"]): os.mkdir(args["graphs_folder"])

model_depth = 10


# randomseed 고정시키기
def seed_everything(seed):
    # tensorflow seed 고정
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # numpy seed 고정
    np.random.seed(seed)

    # pytorch seed 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(args["SEED"])  # seed 고정


train_fight_dir = 'D:/InChang/abnormal_detection_dataset/UBI-FIGHTS/train/fight/'
train_normal_dir = 'D:/InChang/abnormal_detection_dataset/UBI-FIGHTS/train/normal/'
val_fight_dir = 'D:/InChang/abnormal_detection_dataset/UBI-FIGHTS/val/fight/'
val_normal_dir = 'D:/InChang/abnormal_detection_dataset/UBI-FIGHTS/val/normal/'
annotation_dir = 'D:/InChang/abnormal_detection_dataset/UBI-FIGHTS/annotation/'


class CustomDataset(Dataset):
    def __init__(self, video_files, annotation_files, transform=None, FPS=args["FPS"], img_size=args["img_size"], frame_chunk=12):
        self.video_files = video_files
        self.annotation_files = annotation_files
        self.transform = transform
        self.FPS = FPS
        self.img_size = img_size
        self.frame_chunk = frame_chunk
        self.fight_counter = 0

    def __getitem__(self, idx):
        video_frames, video_label = self.get_video_and_label(idx)
        return video_frames, video_label

    def __len__(self):
        return len(self.video_files)

    def get_video_and_label(self, idx):
        video_file = self.video_files[idx]
        annotation_file = self.annotation_files[idx]

        annotation_df = pd.read_csv(annotation_file, header=None)
        fight_indices = np.where(annotation_df.values.squeeze() == 1)[0][::5]

        is_fight = os.path.basename(video_file).startswith('F_')

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f'Error opening video file {video_file}')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = total_frames // self.FPS

        labels = []
        frames = []
        result_imgs = []

        if is_fight:
            indices = fight_indices
            self.fight_counter += len(fight_indices) // self.frame_chunk
        else:
            indices = range(0, annotation_df.shape[0], frame_interval)
            indices = indices[:self.fight_counter * self.frame_chunk]

        for chunk_start in range(0, len(indices), self.frame_chunk):
            chunk_indices = indices[chunk_start:chunk_start + self.frame_chunk]
            for i in chunk_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, frame = cap.read()
                if frame is None:
                    print(f"Error reading frame {i} from video {video_file}")
                    continue
                frame = cv2.resize(frame, self.img_size)  # Resizing the frame before storing
                frame = (frame / 255.).astype(np.float16)
                frames.append(frame)
                labels.append(annotation_df.iloc[i].values[0])

            if len(frames) < self.frame_chunk:
                frames += random.sample(frames, min(self.frame_chunk - len(frames), len(frames)))

            if len(frames) == self.frame_chunk:
                rows = []
                for j in range(0, 12, 3):
                    row_images = frames[j:j+3]
                    concatenated_row = np.hstack(row_images)

                    rows.append(concatenated_row)
                result = np.vstack(rows).astype(np.float32)
                result = cv2.resize(result, (800, 800))
                result_imgs.append(result)
                frames = []
                labels = []

            if len(result_imgs) >= self.FPS:
                break

        cap.release()

        target_label = 1 if any(label == 1 for label in labels) else 0

        return torch.FloatTensor(np.array(result_imgs)).permute(3, 0, 1, 2), torch.tensor([target_label]).float()



train_fight_files = sorted([os.path.join(train_fight_dir, f) for f in os.listdir(train_fight_dir) if os.path.isfile(os.path.join(train_fight_dir, f))])
train_normal_files = sorted([os.path.join(train_normal_dir, f) for f in os.listdir(train_normal_dir) if os.path.isfile(os.path.join(train_normal_dir, f))])
val_fight_files = sorted([os.path.join(val_fight_dir, f) for f in os.listdir(val_fight_dir) if os.path.isfile(os.path.join(val_fight_dir, f))])
val_normal_files = sorted([os.path.join(val_normal_dir, f) for f in os.listdir(val_normal_dir) if os.path.isfile(os.path.join(val_normal_dir, f))])

# annotation 파일 읽어오기
train_fight_annotation_files = [os.path.join(annotation_dir, f + '.csv') for f in [os.path.splitext(os.path.basename(vf))[0] for vf in train_fight_files]]
train_normal_annotation_files = [os.path.join(annotation_dir, f + '.csv') for f in [os.path.splitext(os.path.basename(vf))[0] for vf in train_normal_files]]
val_fight_annotation_files = [os.path.join(annotation_dir, f + '.csv') for f in [os.path.splitext(os.path.basename(vf))[0] for vf in val_fight_files]]
val_normal_annotation_files = [os.path.join(annotation_dir, f + '.csv') for f in [os.path.splitext(os.path.basename(vf))[0] for vf in val_normal_files]]


# 각 클래스의 결과를 합쳐 전체 train, val, test 파일과 주석 획득
train_files = train_fight_files + train_normal_files
train_annotations = train_fight_annotation_files + train_normal_annotation_files

val_files = val_fight_files + val_normal_files
val_annotations = val_fight_annotation_files + val_normal_annotation_files

# test_files = test_fight_files + test_normal_files
# test_annotations = test_fight_annotations + test_normal_annotations

train_dataset = CustomDataset(train_files, train_annotations)
train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args["num_workers"], pin_memory=True)


val_dataset = CustomDataset(val_files, val_annotations,)
val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args["num_workers"], pin_memory=True)

# test_dataset = CustomDataset(test_files, test_annotations)
# test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args["num_workers"], pin_memory=True)

class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0, path=f"{model_save_folder}" + f"model_depth{model_depth}" + f"_early.pth"):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        self.best_score = score

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  val loss is decreased, Saving model ...')
            print('\n')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x_out = self.channel_attention(x)
        x_out = self.spatial_attention(x_out)
        x = x * x_out
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.relu(avg_pool)
        channel_att = self.fc2(avg_pool)
        channel_att = self.sigmoid(channel_att)

        return x * channel_att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        min_pool = torch.min(x, dim=1, keepdim=True)[0]
        concat = torch.cat([max_pool, min_pool], dim=1)
        spatial_att = self.conv(concat)
        spatial_att = self.sigmoid(spatial_att)

        return x * spatial_att


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, expansion=1):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.3)
        self.cbam = CBAM(channels=planes * expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, expansion=4):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(channels=planes * expansion)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)

        out = self.bn3(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.cbam(out)

        out += residual
        out = self.relu(out)
        out = self.dropout(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=args["num_class"]):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.dropout = nn.Dropout(0.3)
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                    nn.Dropout(0.3))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x_conv = self.layer4(x)
        x = self.dropout(x_conv)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x_conv, x


class BaseModel(nn.Module):
    def __init__(self, num_classes=args["num_class"]):
        super(BaseModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(3, 8, (3, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.Dropout(0.3),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 32, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Dropout(0.3),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Dropout(0.3),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Dropout(0.3),
            nn.MaxPool3d((1, 7, 7)),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def train(model, device, args, train_loader, val_loader):
    model.to(device)
    summary(model, (3, 10, 800, 800))
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=args["decay_rate"], amsgrad=True)
    lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.0001, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=True)

    train_acc_for_graph, test_acc_for_graph, train_loss_for_graph, test_loss_for_graph = [], [], [], []
    best_accuracy = 0
    y_pred, y_true = [], []
    y_pred_for_check = []

    early_stopping = EarlyStopping(verbose=True)
    initialize_weights(model)

    for epoch in range(1, args['epoch'] + 1):

        model.train()
        running_corrects = 0

        for videos, labels in tqdm(train_loader, total=len(train_loader), desc='Train', ascii='->=', leave=False):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            _, output = model(videos)
            train_loss = criterion(input=output, target=labels)

            train_loss.backward()
            optimizer.step()

            preds = torch.sigmoid(output) > 0.5
            running_corrects += torch.sum(preds == labels)

            y_pred.append(output.detach().cpu().numpy())
            y_pred_for_check.append(preds.detach().cpu().numpy().astype(int))
            y_true.append(labels.detach().cpu().numpy())

        epoch_acc = running_corrects / (len(train_loader) * args["batch_size"])

        print(f"Epoch {epoch}/{args['epoch']}")
        print(f"train_loss: {train_loss:.5f}   epoch_acc: {epoch_acc:.5f}")

        train_acc_for_graph.append(epoch_acc)
        train_loss_for_graph.append(train_loss)
        model.eval()
        running_corrects = 0
        with torch.no_grad():

            for videos, labels in tqdm(val_loader, total=len(val_loader), desc='Validation', ascii='->=', leave=False):
                videos = videos.to(device)
                labels = labels.to(device)

                _, logit = model(videos)
                test_loss = criterion(input=logit, target=labels)

                valid_preds = torch.sigmoid(logit) > 0.5
                running_corrects += torch.sum(valid_preds == labels)

                y_pred.append(logit.detach().cpu().numpy())
                y_pred_for_check.append(valid_preds.detach().cpu().numpy().astype(int))
                y_true.append(labels.detach().cpu().numpy())

            val_epoch_acc = running_corrects / (len(val_loader) * args["batch_size"])

        print(f'test_loss : {test_loss:.5f}  val_epoch_acc : {val_epoch_acc:.5f}')
        test_acc_for_graph.append(val_epoch_acc)
        test_loss_for_graph.append(test_loss)

        if epoch % 5 == 0:
            lr_decay.step(val_epoch_acc)
            curr_lr = 0
            for params in optimizer.param_groups:
                curr_lr = params['lr']
            print(f"The current learning rate for training is : {curr_lr}")

        if best_accuracy < val_epoch_acc:
            torch.save(model.state_dict(), f"{model_save_folder}" + f"model_depth{model_depth}" + f".pth")
            best_accuracy = val_epoch_acc
            print('Current model has best valid accuracy')

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    y_pred = np.array(y_pred, dtype=np.int64)
    y_pred = y_pred.astype(np.int64)
    y_true = np.array(y_true, dtype=np.int64)

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
    plt.title(f"Receiver Operating Characteristic")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label=f"ResNet{model_depth}+CBAM(AUC) = %0.4f" % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc='lower right')
    plt.savefig(args["graphs_folder"] + 'real' + f"resnet{model_depth}_" + f"auc.png")
    plt.show()

    print('AUC: ', roc_auc)

    train_acc_for_graph = torch.Tensor(train_acc_for_graph).detach().cpu().numpy().tolist()
    test_acc_for_graph = torch.Tensor(test_acc_for_graph).detach().cpu().numpy().tolist()
    train_loss_for_graph = torch.Tensor(train_loss_for_graph).detach().cpu().numpy().tolist()
    test_loss_for_graph = torch.Tensor(test_loss_for_graph).detach().cpu().numpy().tolist()
    print('train_acc_for_graph: ', '\n', train_acc_for_graph)
    print('test_acc_for_graph: ', '\n', test_acc_for_graph)
    print('train_loss_for_graph: ', '\n', train_loss_for_graph)
    print('test_loss_for_graph: ', '\n', test_loss_for_graph)

    plt.plot(train_acc_for_graph)
    plt.plot(test_acc_for_graph)
    plt.title('Train and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.xlim([1, epoch + 1])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"] + 'real' + f"resnet{model_depth}_" + f"acc.png")
    plt.show()

    plt.plot(train_loss_for_graph)
    plt.plot(test_loss_for_graph)
    plt.title('Train and Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([1, epoch + 1])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(args["graphs_folder"] + 'real' + f"resnet{model_depth}_" + f"loss.png")
    plt.show()

    # EER point is where fpr is equal to (1-tpr)
    eer = brentq(lambda x: 1. - x - interp1d(false_positive_rate, true_positive_rate)(x), 0., 1.)
    # Threshold at EER point
    thresh = interp1d(false_positive_rate, thresholds)(eer)
    print('thresh: ', thresh)
    print('EER: {:.5f}, Threshold: {:.5f}'.format(eer, thresh))

    print('train acc', sum(train_acc_for_graph) / len(train_acc_for_graph))
    print('test acc', sum(test_acc_for_graph) / len(test_acc_for_graph))
    print('train loss', sum(train_loss_for_graph) / len(train_loss_for_graph))
    print('test loss', sum(test_loss_for_graph) / len(test_loss_for_graph))


kwargs = {'n_input_channels': 3,
          'conv1_t_size': 7,
          'conv1_t_stride': 1,
          'no_max_pool': False,
          'shortcut_type': 'A',
          'widen_factor': 1.0,
          'n_classes': 1}


def test(model, device, test_loader, weight_file=f"{model_save_folder}" + 'real' + f"model_depth{model_depth}" + f".pth"):
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0
    correct = 0
    criterion = nn.BCEWithLogitsLoss().to(device)

    print("Testing...")
    y_pred, y_true = [], []
    y_pred_for_check = []
    total_loss_list = []
    with torch.no_grad():  # 테스트 시에는 gradient를 계산할 필요가 없으므로
        for videos, labels in tqdm(test_loader, total=len(test_loader), desc='Test', ascii='->=', leave=False):
            videos, labels = videos.to(device), labels.to(device)  # 데이터와 레이블을 device로 이동
            # videos, labels = videos.to(device), labels.unsqueeze(1).float().to(device)  # 데이터와 레이블을 device로 이동

            _, test_output = model(videos)  # 모델의 예측 결과를 가져옴
            loss = criterion(input=test_output, target=labels).item()  # 배치에 대한 손실을 누적
            pred = torch.sigmoid(test_output) > 0.5  # 가장 높은 확률을 가진 인덱스를 가져옴
            correct += pred.eq(labels).sum().item()  # 올바르게 예측한 개수를 누적
            y_pred.append(test_output.detach().cpu().numpy())
            y_pred_for_check.append(pred.detach().cpu().numpy().astype(int))
            y_true.append(labels.detach().cpu().numpy())
            total_loss_list.append(loss)

    # avg_loss = total_loss / len(test_loader.dataset)  # 평균 손실을 계산
    avg_loss = sum(total_loss_list) / len(total_loss_list)  # 평균 손실을 계산
    print(type(y_pred), type(y_true))
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    y_pred = np.array(y_pred, dtype=np.int64)
    y_pred = y_pred.astype(np.int64)
    y_true = np.array(y_true, dtype=np.int64)

    print(type(y_pred), type(y_true))
    print(y_pred.shape, y_true.shape)

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    print(y_pred.shape, y_true.shape)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=np.array(y_pred), average='macro')
    print('test_y_true: ', '\n', y_true[::25])
    print('test_y_pred: ', '\n', y_pred_for_check[::25])
    print('AUC:', auc)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
    plt.title(f"Receiver Operating Characteristic")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label=f"ResNet{model_depth}+CBAM"'(AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.legend(loc='lower right')
    plt.savefig(args["graphs_folder"] + 'real' + f"resnet{model_depth}_test_" + f"auc.png")
    plt.show()

    print('Test AUC: ', roc_auc)
    # EER point is where fpr is equal to (1-tpr)
    eer = brentq(lambda x: 1. - x - interp1d(false_positive_rate, true_positive_rate)(x), 0., 1.)
    # Threshold at EER point
    thresh = interp1d(false_positive_rate, thresholds)(eer)
    print('thresh: ', thresh)
    print('EER: {:.4f}, Threshold: {:.4f}'.format(eer, thresh))
    print('\nTest set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)\n'.format(avg_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    print("Done!")


if model_depth == 10:
    model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
elif model_depth == 18:
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
elif model_depth == 34:
    model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
elif model_depth == 50:
    model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
elif model_depth == 101:
    model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
elif model_depth == 152:
    model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
elif model_depth == 200:
    model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

if __name__ == '__main__':
    train(args=args, device=device, model=model, train_loader=train_loader, val_loader=val_loader)
    # test(model=model, device=device, test_loader=test_loader,
    #      weight_file=f"{model_save_folder}" + 'real' + f"model_depth{model_depth}" + f".pth")

'CUDA_LAUNCH_BLOCKING=1.'