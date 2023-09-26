import os

from .LLFFDataset import LLFFDataset

def get_split_dataset(dataset_type, datadir, want_split='train', training=True, file_lists=['data/Balloon2',], **kwargs):
    """
    Retrieved desired dataset class
    :param dataset_type dataset type name (srn|dvr|dvr_gen, etc)
    :param datadir root directory name for the dataset. For SRN/multi_obj data:
    if data is in dir/cars_train, dir/cars_test, ... then put dir/cars
    :param want_split root directory name for the dataset
    :param training set to False in eval scripts
    """
    dset_class, train_aug = None, None
    flags, train_aug_flags = {}, {}

    if dataset_type.startswith('monocular'):
        flags['factor'] = 2
        flags['frame2dolly'] = -1
        flags['spherify'] = False
        flags['num_novelviews'] = 60
        flags['z_trans_multiplier'] = 5.
        flags['x_trans_multiplier'] = 1.
        flags['y_trans_multiplier'] = 0.33
        flags['no_ndc'] = False
        flags['file_lists'] = file_lists
        dset_class = LLFFDataset
        train_aug = None
    else:
        raise NotImplementedError('Unsupported dataset type', dataset_type)

    want_train = want_split != 'val' and want_split != 'test'
    want_val = want_split != 'train' and want_split != 'test'
    want_test = want_split != 'train' and want_split != 'val'

    if want_train:
        train_set = dset_class(datadir, stage='train', **flags, **kwargs)
        if train_aug is not None:
            train_set = train_aug(train_set, **train_aug_flags)

    if want_val:
        val_set = dset_class(datadir, stage='val', **flags, **kwargs)

    if want_test:
        test_set = dset_class(datadir, stage='test', **flags, **kwargs)

    if want_split == 'train':
        return train_set, None, None
    elif want_split == 'val':
        return val_set
    elif want_split == 'test':
        return test_set
    return train_set, val_set, test_set
