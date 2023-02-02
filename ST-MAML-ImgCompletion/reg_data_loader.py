
from collections import namedtuple
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
Task = namedtuple('Task', ['x', 'y', 'task_info'])
from utils import *








class get_dloader():
    def __init__(self, root, dataset, meta_batch_size, n_spt, img_size, train):
        self._dataset = dataset
        self.n_spt = n_spt
        self._root = root
        self.name = dataset
        self._meta_batch_size = meta_batch_size
        self.input_size = n_spt
        self.output_size = img_size*img_size
        self.img_size = img_size
        self.train=train
        self.transform = self.comp_transform()
        self.dset = self.get_dataset()
        self._dloader = self.dloader()
        


    def comp_transform(self):
        transform = transforms.Compose(
            [transforms.Resize(self.img_size),
             transforms.ToTensor()]
        )
        return transform

    def get_dataset(self):
        if self.name == 'mnist':
            dset = datasets.MNIST(root=self._root, download=True, train=self.train, transform=self.transform)
        elif self.name == 'fmnist':
            dset = datasets.FashionMNIST(root=self._root, download=True, train=self.train, transform=self.transform)
        elif self.name == 'kmnist':
            dset = datasets.KMNIST(root=self._root, download=True, train=self.train, transform=self.transform)
        return dset

    def dloader(self):
        my_collator = MyCollator(self.n_spt, self.output_size)
        my_sampler = EpisodicBatchSampler(10000000, len(self.dset), self._meta_batch_size)
        dloader = DataLoader(self.dset, batch_sampler=my_sampler, collate_fn=my_collator, num_workers=0, pin_memory=True)
        return dloader

    def __iter__(self):
        for spt_x, spt_y, qry_x, qry_y in iter(self._dloader):
            train_tasks = [Task(spt_x_i, spt_y_i, self.name) for spt_x_i, spt_y_i in zip(spt_x, spt_y)]
            val_tasks = [Task(qry_x_i, qry_y_i, self.name) for qry_x_i, qry_y_i in zip(qry_x, qry_y)]
            yield train_tasks, val_tasks



class EpisodicBatchSampler(object):
    def __init__(self, n_episodes, n_data, n_batchsize):
        self.n_data = n_data
        self.n_bs = n_batchsize
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_data)[:self.n_bs] 






class MultimodalFewShotDataset(object):

    def __init__(self, datasets, num_total_batches,
                 name='MultimodalFewShot',
                 mix_meta_batch=True, mix_mini_batch=False,
                 train=True, verbose=False, txt_file=None):
        self._datasets = datasets
        self._num_total_batches = num_total_batches
        self.name = name
        self.num_dataset = len(datasets)
        self.dataset_names = [dataset.name for dataset in self._datasets]
        self._meta_batch_size = datasets[0]._meta_batch_size

        self._train = train
        self._verbose = verbose


        # make sure all input/output sizes match
        input_size_list = [dataset.input_size for dataset in self._datasets]
        assert input_size_list.count(input_size_list[0]) == len(input_size_list)
        output_size_list = [dataset.output_size for dataset in self._datasets]
        assert output_size_list.count(output_size_list[0]) == len(output_size_list)
        self.input_size = datasets[0].input_size
        self.output_size = datasets[0].output_size

        # build iterators
        self._datasets_iter = [iter(dataset) for dataset in self._datasets]
        self._iter_index = 0
        self._mix_meta_batch = mix_meta_batch
        self._mix_mini_batch = mix_mini_batch

        # print info
        print('Multimodal Few Shot Datasets: {}'.format(' '.join(self.dataset_names)))
        print('mix meta batch: {}'.format(mix_meta_batch))
        print('mix mini batch: {}'.format(mix_mini_batch))

    def __next__(self):
        if self.n < self._num_total_batches:
            all_train_tasks = []
            all_val_tasks = []
            for dataset_iter in self._datasets_iter:
                train_tasks, val_tasks = next(dataset_iter)
                all_train_tasks.extend(train_tasks)
                all_val_tasks.extend(val_tasks)

            if not self._mix_mini_batch:
                # mix them to obtain a meta batch
                """
                # randomly sample task
                dataset_indexes = np.random.choice(
                    len(all_train_tasks), size=self._meta_batch_size, replace=False)
                """
                # balancedly sample from all datasets
                dataset_indexes = []
                if self._train:
                    dataset_start_idx = np.random.randint(0, self.num_dataset)
                else:
                    dataset_start_idx = (self._iter_index + self._meta_batch_size) % self.num_dataset
                    self._iter_index += self._meta_batch_size
                    self._iter_index = self._iter_index % self.num_dataset

                for i in range(self._meta_batch_size):
                    dataset_indexes.append(
                        np.random.randint(0, self._meta_batch_size) +
                        ((i + dataset_start_idx) % self.num_dataset) * self._meta_batch_size)

                train_tasks = []
                val_tasks = []
                dataset_names = []
                for dataset_index in dataset_indexes:
                    train_tasks.append(all_train_tasks[dataset_index])
                    val_tasks.append(all_val_tasks[dataset_index])
                    dataset_names.append(self._datasets[dataset_index // self._meta_batch_size].name)
                if self._verbose:
                    print('Sample from: {} (indexes: {})'.format(
                        [name for name in dataset_names], dataset_indexes))
                self.n += 1
                return train_tasks, val_tasks
            else:
                # mix them to obtain a mini batch and make a meta batch
                raise NotImplementedError
        else:
            raise StopIteration

    def __iter__(self):
        self.n = 0
        return self
