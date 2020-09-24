import datetime

import torch


class ModelEnvironment:

    def __init__(self, args):
        self.args = args
        self.best_epoch = None
        self.epochs = []
        self.test_evaluations = []

    def add_test_result(self, epoch, dataset_name, result):
        filename = f'test_{dataset_name}_{datetime.datetime.now().strftime(f"%Y%m%d_%H%M%S")}.bin'
        path = self.args.path / 'epochs' / str(epoch)
        path.resolve().mkdir(parents=True, exist_ok=True)
        path = path / filename
        torch.save(result, path)
        self.test_evaluations.append({
            'dataset_name': dataset_name,
            'epoch': epoch,
            'path': path,
            'stats': result['stats']
        })
