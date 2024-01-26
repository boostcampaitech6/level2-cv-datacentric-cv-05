import argparse

class Parser(object):
    """
    명령줄 인수를 파싱하기 위한 클래스.

    이 클래스는 머신러닝 또는 딥러닝 실험의 구성 설정을 위한 명령줄 인수를 정의하고 관리합니다. 
    사용자는 이 클래스를 통해 실험에 필요한 다양한 설정값을 지정할 수 있습니다.

    Attributes:
        description (str): 명령줄 파서에 대한 설명. 기본값은 빈 문자열입니다.
        parser (argparse.ArgumentParser): 명령줄 인수를 파싱하는데 사용되는 argparse의 Parser 객체.

    Methods:
        create_parser: 명령줄 인자를 파서에 추가하는 메서드. 다양한 설정 및 데이터 경로 인자들이 포함됩니다.
    """
    def __init__(self, description=""):
        self.description = description
        self.parser = argparse.ArgumentParser(description=self.description)
    
    def create_parser(self):        
        self.parser.add_argument('--config', default='./config/base.yml', help='Path to the configuration file in YAML format.')
        self.parser.add_argument('--save-dir', default="./results", help="Directory for saving training outputs like models and logs.")
        self.parser.add_argument('--output-dir', default="./output", help="Directory to store additional output files.")
        self.parser.add_argument('--project-name', default="exp", help="Name of the Weights & Biases project for tracking experiments.")
        self.parser.add_argument('--exp-name', default="exp", help="Name of this specific experiment run.")
        self.parser.add_argument('--test-exp-num', default=None, help="Experiment number for the test run.")
        
        # dataset & augmentation
        self.parser.add_argument('--data_dir', type=str, default= '../data/medical')
        self.parser.add_argument('--json_path', type=str, default='split/train_fold_0.json', help="Enter data_dir as the relative path of the variable.")
        self.parser.add_argument('--val_json_path', type=str, default='split/val_fold_0.json', help="Enter data_dir as the relative path of the variable.")
        self.parser.add_argument('--num_workers', type=int, default=8)
        self.parser.add_argument('--image_size', type=int, default=2048)
        self.parser.add_argument('--input_size', type=int, default=1024)
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--dataset', default="SceneTextDataset", help="Type of dataset to be used (e.g., 'TestDataset').")
        self.parser.add_argument('--augmentation', default="BaseAugmentation", help="Augmentation method to be applied to the dataset.")
        self.parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
        self.parser.add_argument('--with_pickle', default=False, type=bool)
        self.parser.add_argument('--pickle_path', type=str, default='split/train_fold_0.pkl', help="Enter $data_dir/ufo as the relative path of the variable.")
        
        # training settings
        self.parser.add_argument('--seed', default=207, help="Seed for random number generation to ensure reproducibility.")
        self.parser.add_argument('--max-epochs', default=150, help="Maximum number of training epochs.")
        self.parser.add_argument('--optimizer', default="Adam", help="Optimization algorithm to be used.")
        self.parser.add_argument('--lr', default=1e-3, help="Learning rate for the optimizer.")
        self.parser.add_argument('--scheduler', default="MultiStepLR", help="Learning rate scheduler to be used.")
        self.parser.add_argument('--lr-decay-step', default=75, help="Step size for learning rate decay.")
        
    def print_args(self, args):
        print("Arguments:")
        for arg in vars(args):
            print("\t{}: {}".format(arg, getattr(args, arg)))
