import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Check the shape of the input tensor
        #print(x.shape)

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        #print(avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=2, kernel_size=7):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.channel_attention = self.channel_attention.to('cuda')
        self.spatial_attention = SpatialAttention(kernel_size)
        self.spatial_attention = self.spatial_attention.to('cuda')

    def forward(self, x):
        channel_out = self.channel_attention(x)
        x1= x*channel_out
        spatial_out = self.spatial_attention(x1)
        return  spatial_out*x


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):

        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform

        if args.dataset_name=='iu_xray':


             self.ann = json.loads(open(self.ann_path, 'r').read())

             self.examples = self.ann[self.split]

             for i in range(len(self.examples)):
                 self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
                 self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])


        elif args.dataset_name=='mimic_cxr':
            self.ann = json.loads(open(self.ann_path, 'r').read())
            #print(self.split)
            self.examples = self.ann[self.split]
            #print(*self.examples)
            for i in range(len(self.examples)):
                self.examples[i]['ids'] = tokenizer(self.examples[i]['findings'])[:self.max_seq_length]
                self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])


            """"#print(self.split)
            if self.split =='train':
                self.ann = json.loads(open(self.ann_path1, 'r').read())
                # print(self.ann[self.split])
                self.examples1 = self.ann['reports']
                #print(self.ann['reports'])
                for i in range(len(self.examples1)):
                    self.examples1[i]['ids'] = tokenizer(self.examples1[i]['findings'])[:self.max_seq_length]
                    self.examples1[i]['mask'] = [1] * len(self.examples1[i]['ids'])
                    self.examples1[i]['split'] = 'train'

            elif self.split =='val':
                print(self.ann[self.split])
                self.ann = json.loads(open(self.ann_path2, 'r').read())
                # print(self.ann[self.split])
                self.examples2 =self.examples2.append(self.ann['reports'])
                for i in range(len(self.examples2)):
                    self.examples2[i]['ids'] = tokenizer(self.examples2[i]['findings'])[:self.max_seq_length]
                    self.examples2[i]['mask'] = [1] * len(self.examples2[i]['ids'])
                    self.examples2[i]['split'] = 'valid'

            elif self.split == 'test':
                self.ann = json.loads(open(self.ann_path3, 'r').read())
                # print(self.ann[self.split])
                self.examples3 = self.ann['reports']
                for i in range(len(self.examples3)):
                    self.examples3[i]['ids'] = tokenizer(self.examples3[i]['findings'])[:self.max_seq_length]
                    self.examples3[i]['mask'] = [1] * len(self.examples3[i]['ids'])
                    self.examples3[i]['split'] = 'test'

            #print(self.examples1)
            self.examples = [*self.examples1, self.examples2, self.examples3]"""



    def __len__(self):
       return len(self.examples)





class IuxrayMultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(IuxrayMultiImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)
        #print(self.label)

    def _load_data(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['id']
            #print(idx)
            label = row[1:].to_list()

            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))
        #print(label_dict)

        return label_dict

    def __getitem__(self, idx):
        example = self.examples[idx]
        #print(self.examples[idx])

        image_id = example['id']

        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        image = torch.stack((image_1, image_2), 0)
        """if image.size(0) == 8:
            # 创建 AttentionModule 对象
            in_planes = 8  # 根据你的实际情况设置输入通道数
            attention_module = AttentionModule(in_planes)
            # input_tensor = torch.randn(1, in_planes, 32, 32)  # 根据你的实际情况设置输入张量的形状
            print(image.shape)
            image = attention_module(image)
        else:
            print(image.shape)
            print('no operation')"""
        #print(image.shape)
        report_ids = example['ids']
        report_masks = example['mask']
        #print(report_masks)
        seq_length = len(report_ids)

        pid = image_id
        #print(pid)
        try:
            labels = torch.tensor(self.label[pid], dtype=torch.float32)
            #print(labels)
        except:
            # print('Except id ', pid)
            labels = torch.tensor([0 for _ in range(14)], dtype=torch.float32)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        #print(self.examples)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(MimiccxrSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = str(int(row['study_id']))
            #print(idx)
            label = row[2:].to_list()

            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))
        #print(label_dict)

        return label_dict

    def __getitem__(self, idx):
        #print(self.examples[idx])
        example = self.examples[idx]
        image_id = example['rep_id']
        image_path = example['images']
        #print(image_path[0]['img_name'])
        image = Image.open(os.path.join(self.image_dir, image_path[0]['img_name']+'.jpg')).convert('RGB')
        #print(image)
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        #d = self.label[self.label['study_id'] == image_id]

        #labels = torch.tensor(d.values.tolist()[0][8:], dtype=torch.float32)
        pid = image_id
        #print(type(pid))

        try:
            labels = torch.tensor(self.label[pid], dtype=torch.float32)
            #print(labels)
        except:
            # print('Except id ', pid)
            labels = torch.tensor([0 for _ in range(14)], dtype=torch.float32)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CovidSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(CovidSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_file):
        labels = {}

        # print(f"Loading data from {label_file}")

        data = pd.read_csv(label_file)
        # data = data[data['split'] == self.subset]
        for index, row in data.iterrows():
            idx = row['idx']
            label = [1, 0] if row['label'] == '轻型' else [0, 1]
            labels[idx] = label

        return labels

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = torch.tensor(self.label[image_id], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CovidAllImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = torch.tensor(example['label'], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample