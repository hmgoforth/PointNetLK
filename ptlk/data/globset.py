""" glob. """
import os
import glob
import copy
import six
import numpy as np
import torch
import torch.utils.data


def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def classes_to_cinfo(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []
    #class_size = [0 for i in range(len(class_to_idx))]
    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue

        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue

        #count = 0
        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
                #count += 1
        #class_size[target_idx] = count

    return samples


class Globset(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """
    def __init__(self, rootdir, pattern, fileloader, transform=None, classinfo=None):
        super().__init__()

        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        self.rootdir = rootdir
        self.pattern = pattern
        self.fileloader = fileloader
        self.transform = transform

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.rootdir)
        fmt_str += '    File Patterns: {}\n'.format(self.pattern)
        fmt_str += '    File Loader: {}\n'.format(self.fileloader)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.fileloader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def num_classes(self):
        return len(self.classes)

    def class_name(self, cidx):
        return self.classes[cidx]

    def indices_in_class(self, cidx):
        targets = np.array(list(map(lambda s: s[1], self.samples)))
        return np.where(targets == cidx).tolist()

    def select_classes(self, cidxs):
        indices = []
        for i in cidxs:
            idxs = self.indices_in_class(i)
            indices.extend(idxs)
        return indices

    def split(self, rate):
        """ dateset -> dataset1, dataset2. s.t.
            len(dataset1) = rate * len(dataset),
            len(dataset2) = (1-rate) * len(dataset)
        """
        orig_size = len(self)
        select = np.zeros(orig_size, dtype=int)
        csize = np.zeros(len(self.classes), dtype=int)
        dsize = np.zeros(len(self.classes), dtype=int)

        for i in range(orig_size):
            _, target = self.samples[i]
            csize[target] += 1
        dsize = (csize * rate).astype(int)
        for i in range(orig_size):
            _, target = self.samples[i]
            if dsize[target] > 0:
                select[i] = 1
                dsize[target] -= 1

        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        samples1 = list(map(lambda i: dataset1.samples[i], np.where(select == 1)[0]))
        samples2 = list(map(lambda i: dataset2.samples[i], np.where(select == 0)[0]))

        dataset1.samples = samples1
        dataset2.samples = samples2
        return dataset1, dataset2



#EOF
