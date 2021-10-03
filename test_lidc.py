import torch
import argparse
from train_bayesian import Bayeisan_Trainer
import numpy as np
def main():
    parser = argparse.ArgumentParser(description="PyTorch Bayesian UNet Test")
    parser.add_argument('--save-path', type=str, default='run')
    parser.add_argument('--is-train', action='store_true', default=False,
                        help='Training script for Testing script')
    parser.add_argument('--dataset', type=str, default='uncertain-brats',
                        choices=['brats', 'uncertain-brats', 'uncertain-brain-growth', 'uncertain-kidney',
                                 'uncertain-prostate', 'lidc', 'lidc-rand'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=2,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--loss-type', type=str, default='ELBO',
                        choices=['soft-dice', 'dice', 'fb-dice', 'ce', 'level-thres', "ELBO"],
                        help='loss func type (default: ce)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='batten_unet',
                        help='set the checkpoint name')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    parser.add_argument('--nclass', type=int, default=6,
                        help='number of texture classes for training texture segmentation model')

    parser.add_argument('--model', type=str, default='multi-bunet',
                        help='specify the model, default by unet',
                        choices=['unet', 'prob-unet', 'multi-unet', 'decoder-unet', 'attn-unet', 'pattn-unet',
                                 'pattn-unet-al', "batten-unet", "multi-bunet", "multi-atten-bunet"])
    parser.add_argument('--pretrained', type=str, default=None,
                        help='specify the path to pretrained model parameters')

    parser.add_argument('--nchannels', type=int, default=4, help='set input channel of the model')

    parser.add_argument('--dropout', action='store_true', default=False, help='add drop out to the model')

    parser.add_argument('--drop-p', type=float, default=0.5, help='probability of applying dropout')

    parser.add_argument('--task-num', type=int, default=1, help='task No. for uncertain dataset')
    parser.add_argument('--num-sample', type=int, default=50, help="Sampling number")
    parser.add_argument("--model-path", type = str, default= None, help = "The path of the model that you want to test")
    parser.add_argument("--beta-type", action='store_const', default= 'standard', const='standard',
                        help="the beta type default valu")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    print(args)
    torch.cuda.cudann_enable = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    trainer = Bayeisan_Trainer(args)
    if args.model_path is None:
        raise ValueError("Argument --model_path must not be None value")

    trainer.test(args.model_path)

    trainer.writer.close()
if __name__ == "__main__":
    main()