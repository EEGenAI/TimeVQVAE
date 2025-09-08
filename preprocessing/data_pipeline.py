from torch.utils.data import DataLoader
from .preprocess_ucr import DatasetImporterUCR, UCRDataset, CustomDataset, DatasetImporterCustom


def build_data_pipeline(batch_size, dataset_importer: DatasetImporterUCR, config: dict, kind: str) -> DataLoader:
    """
    :param config:
    :param kind train/valid/test
    """
    num_workers = config['dataset']["num_workers"]

    # DataLoader
    if kind == 'train':
        train_dataset = UCRDataset("train", dataset_importer)
        # `drop_last=False` due to some datasets with a very small dataset size.
        return DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)
    elif kind == 'test':
        test_dataset = UCRDataset("test", dataset_importer)
        return DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)
    else:
        raise ValueError


def build_custom_data_pipeline(batch_size, dataset_importer: DatasetImporterCustom, config: dict, kind: str) -> DataLoader:
    """
    :param config:
    :param kind train/valid/test
    """
    num_workers = config['dataset']["num_workers"]

    # DataLoader
    if kind == 'train':
        custom_dataset = CustomDataset('train', dataset_importer)
        # `drop_last=False` due to some datasets with a very small dataset size.
        return DataLoader(custom_dataset, batch_size, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)
    elif kind == 'test':
        custom_dataset = CustomDataset('test', dataset_importer)
        # `drop_last=False` due to some datasets with a very small dataset size.
        return DataLoader(custom_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)
    else:
        raise ValueError
