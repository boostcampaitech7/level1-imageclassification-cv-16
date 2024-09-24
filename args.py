import argparse
from argparse import Namespace
from argparse import ArgumentParser
class Custom_arguments_parser:
    def __init__(self, mode):
        self.mode = mode
    
    def get_base_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', type=str, default='train', help='Select mode train or test default is train', action='store')
        parser.add_argument('--device', type=str, default='cpu', help='Select device to run, default is cpu', action='store')

        parser.add_argument('--data_root', type=str, default='./data', help='Path to data root', action='store')
        parser.add_argument('--csv_path', type=str, default='./data/train.csv', help='Path to csv', action='store')
    
        parser.add_argument('--transform_type', type=str, default='albumentations', help='Select transform type, default is albumentation', action='store')
        parser.add_argument('--augmentations', type=str, default="hflip_vflip_rotate", help='Select augmentations to use, default is hflip_vflip_rotate', action='store')
        parser.add_argument('--adjust_ratio', help='Turn True to adjust the ratio', action='store_true')
    
        parser.add_argument('--height', type=int, default=224, help="Select input img height, default is 224", action='store')
        parser.add_argument('--width', type=int, default=224, help="Select input img width, default is 224", action='store')
    
        parser.add_argument('--num_classes', type=int, default=500, help="Select number of classes, default is 500", action='store')
    
        parser.add_argument('--model', type=str, default='cnn', help='Select a model to train, default is cnn', action='store')
        parser.add_argument('--batch', type=int, default=64, help='Select batch_size, default is 64', action='store')

        parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint, default is None', action='store')

        parser.add_argument('--verbose', help='add --verbose to turn off progress bar display', action='store_true')

        return parser

    def get_parser(self) -> Namespace:
        if self.mode == 'train':
            return self.parse_train_args_and_config()
        elif self.mode == 'test':
            return self.parse_test_args_and_config()
        else:
            raise Exception('missing/incorrect mode value. please choose between train or test')
    
    def parse_base_args(self) -> Namespace:
        return self.get_base_args().parse_args()

    ## for train.py
    def parse_train_args_and_config(self) -> Namespace:
        parser = self.get_base_args()
        
        parser.add_argument('--val_csv', type=str, default='./data/val.csv', help='Path to val csv', action='store')
        
        parser.add_argument('--auto_split', type=bool, default=True, help='Set auto_split, requires train & val csv if False', action='store')
        parser.add_argument('--split_seed', type=int, default=42, help='Set split_seed, default is 42', action='store')
        parser.add_argument('--stratify', type=str, default='target', help='Set balance split', action='store')
    
        parser.add_argument('--lr', type=float, default=0.01, help='Select Learning Rate, default is 0.01', action='store')
        parser.add_argument('--lr_scheduler', type=str, default="stepLR", help='Select LR scheduler, default is stepLR', action='store')
        parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1, help='Select LR scheduler gamma, default is 0.1', action='store')
        parser.add_argument('--lr_scheduler_epochs_per_decay', type=int, default=2, help='Select LR scheduler epochs_per_decay, default is 2', action='store')
        parser.add_argument('--loss', type=str, default='CE', help='Select Loss, default is Cross Entropy(CE)', action='store')
        parser.add_argument('--optim', type=str, default='adam', help='Select a optimizer, default is adam', action='store')
        parser.add_argument('--epochs', type=int, default='100', help='Select total epochs to train, default is 100 epochs', action='store')
        parser.add_argument('--r_epochs', type=int, default='2', help='Select total data swap epochs, default is last 2 epochs', action='store')
        parser.add_argument('--seed', type=int, default=2024, help='Select seed, default is 2024', action='store')

        parser.add_argument('--resume', help='resuming training, default is False meaning new training (requires weights_path for checkpoints)', action='store_true')
        parser.add_argument('--early_stopping', type=int, default=10, help='Select number of epochs to wait for early stoppoing', action='store')
    
        return parser.parse_args()
    
    ## for test.py
    def parse_test_args_and_config(self) -> Namespace:
        parser = self.get_base_args()
        
        parser.add_argument('--output_path', type=str, default='output.csv', help='Path for csv result', action='store')
        
        return parser.parse_args()

# if __name__=='__main__':
    
#     asd = arguments_parser(mode='train')
    
#     a = asd.parse_base_args()
    
#     print(a)