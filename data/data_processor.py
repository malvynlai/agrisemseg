from torchvision import transforms
from torchvision import datasets
from PIL import Image
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from extraction.class_dataset import AgriDataset
from sklearn.model_selection import train_test_split
import datetime
import os
from data.loader import load_dataset

class AgricultureConfigs:
    def __init__(self, model='none', dataset='Agriculture', bands=['NIR', 'RGB'], k=0, k_folder=0, note=''):
        self.model = model
        self.dataset = dataset
        self.bands = bands
        self.loader = AgriDataset  
        self.labels = 9 
        self.nb_classes = len(9)
        self.weights = []

        self.k_folder = k_folder
        self.k = k
        self.input_size = [512, 512]
        self.scale_rate = 1.0
        self.val_size = [512, 512]
        self.train_samples = 12901
        self.val_samples = 1200
        self.train_batch = 7
        self.val_batch = 7

        self.pre_norm = False
        self.seeds = 69278

        self.lr = 1.5e-4 / np.sqrt(3.0)
        self.lr_decay = 0.9
        self.max_iter = 1e8

        self.weight_decay = 2e-5
        self.momentum = 0.9

        self.ckpt_path = '../ckpt'
        self.snapshot = ''

        self.print_freq = 100
        self.save_pred = False
        self.save_rate = 0.1
        self.best_record = {}

        self.suffix_note = note

        self._create_directories()

    def split_train_val_test_sets(name, bands, KF, k, seeds):
        np.random.seed(seeds)
        dataset = load_dataset(name, bands)
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        if KF:
            fold_size = len(indices) // KF
            val_indices = indices[k * fold_size:(k + 1) * fold_size]
            train_indices = np.concatenate((indices[:k * fold_size], indices[(k + 1) * fold_size:]))
        else:
            train_size = int(0.8 * len(indices))
            val_size = int(0.1 * len(indices))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

        train_dict = {i: dataset[i] for i in train_indices}
        val_dict = {i: dataset[i] for i in val_indices}
        test_dict = {i: dataset[i] for i in test_indices}

        return train_dict, val_dict, test_dict

    def _create_directories(self):
        check_mkdir(self.ckpt_path)
        check_mkdir(os.path.join(self.ckpt_path, self.model))

        bandstr = '-'.join(self.bands)
        if self.k_folder:
            subfolder = f"{self.dataset}_{bandstr}_kf-{self.k_folder}-{self.k}"
        else:
            subfolder = f"{self.dataset}_{bandstr}"
        if self.suffix_note:
            subfolder += f"-{self.suffix_note}"

        check_mkdir(os.path.join(self.ckpt_path, self.model, subfolder))
        self.save_path = os.path.join(self.ckpt_path, self.model, subfolder)

    def get_file_list(self):
        return self.split_train_val_test_sets(name=self.dataset, bands=self.bands, KF=self.k_folder, k=self.k, seeds=self.seeds)

    def get_dataset(self):
        train_dict, val_dict, test_dict = self.get_file_list()
        train_set = self.loader(mode='train', file_lists=train_dict, pre_norm=self.pre_norm,
                                num_samples=self.train_samples, windSize=self.input_size, scale=self.scale_rate)
        val_set = self.loader(mode='val', file_lists=val_dict, pre_norm=self.pre_norm,
                              num_samples=self.val_samples, windSize=self.val_size, scale=self.scale_rate)
        return train_set, val_set

    def resume_train(self, net):
        if not self.snapshot:
            curr_epoch = 1
            self.best_record = {'epoch': 0, 'val_loss': 0, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0, 'f1': 0}
            print('Training from scratch.')
        else:
            print(f'Training resumes from {self.snapshot}')
            net.load_state_dict(torch.load(os.path.join(self.save_path, self.snapshot)))
            split_snapshot = self.snapshot.split('_')
            curr_epoch = int(split_snapshot[1]) + 1
            self.best_record = {
                'epoch': int(split_snapshot[1]),
                'val_loss': float(split_snapshot[3]),
                'acc': float(split_snapshot[5]),
                'acc_cls': float(split_snapshot[7]),
                'mean_iu': float(split_snapshot[9]),
                'fwavacc': float(split_snapshot[11]),
                'f1': float(split_snapshot[13])
            }
        return net, curr_epoch

    def print_best_record(self):
        print(
            f"[best_{self.best_record['epoch']}]: [val loss {self.best_record['val_loss']:.5f}], "
            f"[acc {self.best_record['acc']:.5f}], [acc_cls {self.best_record['acc_cls']:.5f}], "
            f"[mean_iu {self.best_record['mean_iu']:.5f}], [fwavacc {self.best_record['fwavacc']:.5f}], "
            f"[f1 {self.best_record['f1']:.5f}]"
        )

    def update_best_record(self, epoch, val_loss, acc, acc_cls, mean_iu, fwavacc, f1):
        print('----------------------------------------------------------------------------------------')
        print(
            f"[epoch {epoch}]: [val loss {val_loss:.5f}], [acc {acc:.5f}], [acc_cls {acc_cls:.5f}], "
            f"[mean_iu {mean_iu:.5f}], [fwavacc {fwavacc:.5f}], [f1 {f1:.5f}]"
        )
        self.print_best_record()
        print('----------------------------------------------------------------------------------------')
        if mean_iu > self.best_record['mean_iu'] or f1 > self.best_record['f1']:
            self.best_record.update({
                'epoch': epoch,
                'val_loss': val_loss,
                'acc': acc,
                'acc_cls': acc_cls,
                'mean_iu': mean_iu,
                'fwavacc': fwavacc,
                'f1': f1
            })
            return True
        return False

    def display(self):
        """Print out all configuration values."""
        print("\nConfigurations:")
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                print(f"{attr:30} {getattr(self, attr)}")
        print("\n")

    def write2txt(self):
        with open(os.path.join(self.save_path, f"{datetime.datetime.now()}.txt"), 'w') as file:
            for attr in dir(self):
                if not attr.startswith("__") and not callable(getattr(self, attr)):
                    line = f"{attr:30} {getattr(self, attr)}"
                    file.write(line + '\n')


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
