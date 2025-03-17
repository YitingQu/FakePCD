import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Detection')
    parser.add_argument('--shapes', default=None, nargs='+', help='shape name str or shape name list')
    parser.add_argument('--sources', default=None, nargs='+', help='generative model name str or generative model name list')
    parser.add_argument('--data_dir', default='FakeCloud/DGCNN/exp_data')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        help='Model to use')
    parser.add_argument('--batch_size', type=int, default=20, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--epochs_contrastive', type=int, default=100, metavar='N',
                        help='number of episode to train contrastively ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use'),
    parser.add_argument('--percentile', type=int, default=None,
                        help='percentile to control threshold')
    parser.add_argument('--threshold', type=float, default=None,
                        help='if not use percentile and select threshold directly')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--proj_dim', type=int, default=128, metavar='N',
                        help='Dimension of projection head')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='eval model path')
    parser.add_argument('--contrastive_checkpoint', type=str, default='', metavar='N',
                        help='contrastive checkpoint')
    parser.add_argument('--pretrained_path', type=str, default='', metavar='N',
                        help='pretrained model path (encoder)')
    parser.add_argument('--save_freq', type=int, default=20, help='save model every N epochs')
    parser.add_argument('--save_dir', type=str, default='', help='base directory for saving results')
    parser.add_argument('--resume_model', default=None, help='resume checkpoint path')

    # regarding robustness evaluation, only valid for testing set

    parser.add_argument('--translate', type=bool, default=False,
                        help='perturbation: translate')
    parser.add_argument('--jitter', type=bool, default=False,
                        help='perturbation: jitter')
    parser.add_argument('--rotate', type=bool, default=False,
                        help='perturbation: rotate')
    parser.add_argument('--translation_value', type=float, default=None, metavar='N',
                        help='origin moving value')
    parser.add_argument('--sigma', type=float, default=None, metavar='N',
                        help='gaussian noise to add')
    parser.add_argument('--theta', type=float, default=None, metavar='N',
                        help='rotation_angle')

    args = parser.parse_args()
    return args
