from __future__ import print_function

import matplotlib; matplotlib.use('Agg')
import os
import os.path as osp
import argparse

from train import train 
from test import test
from test_beam import test_beam 
  
parser = argparse.ArgumentParser(description='PyTorch Convolutional Image Captioning Model')

parser.add_argument('model_dir', help='output directory to save models & results')

parser.add_argument('-g', '--gpu', type=int, default=0,\
                    help='gpu device id')

parser.add_argument('--coco_root', type=str, default= './data/coco/',\
                    help='directory containing coco dataset train2014, val2014, & annotations')

parser.add_argument('-t', '--is_train', type=int, default=1,\
                    help='use 1 to train model')

parser.add_argument('-e', '--epochs', type=int, default=30,\
                    help='number of training epochs')

parser.add_argument('-b', '--batchsize', type=int, default=20,\
                    help='number of images per training batch')

parser.add_argument('-c', '--ncap_per_img', type=int, default=5,\
                    help='ground-truth captions per image in training batch')

parser.add_argument('-n', '--num_layers', type=int, default=3,\
                    help='depth of convcap network')

parser.add_argument('-m', '--nthreads', type=int, default=4,\
                    help='pytorch data loader threads')

parser.add_argument('-ft', '--finetune_after', type=int, default=8,\
                    help='epochs after which vgg16 is fine-tuned')

parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5,\
                    help='learning rate for convcap')

parser.add_argument('-st', '--lr_step_size', type=int, default=15,\
                    help='epochs to decay learning rate after')

parser.add_argument('-sc', '--score_select', type=str, default='CIDEr',\
                    help='metric to pick best model')

parser.add_argument('--beam_size', type=int, default=1, \
                    help='beam size to use for test') 

parser.add_argument('--attention', dest='attention', action='store_true', \
                    help='Use this for convcap with attention (by default set)')

parser.add_argument('--no-attention', dest='attention', action='store_false', \
                    help='Use this for convcap without attention')

parser.set_defaults(attention=True)

args = parser.parse_args()


def main():
  """Train model and run inference on coco test set to output metrics"""

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

  if(args.is_train == 1):
    train(args)

  bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
  if(osp.exists(bestmodelfn)):

    if(args.beam_size == 1):
      scores = test(args, 'test', modelfn=bestmodelfn)
    else:
      scores = test_beam(args, 'test', modelfn=bestmodelfn)

    print('TEST set scores')
    for k, v in scores[0].iteritems():
      print('%s: %f' % (k, v))
  else:
    raise Exception('No checkpoint found %s' % bestmodelfn)

if __name__ == '__main__': 
  main()
