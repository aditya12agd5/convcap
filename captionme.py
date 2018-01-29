from __future__ import print_function

import matplotlib; matplotlib.use('Agg')
import os
import os.path as osp
import argparse

import numpy as np
import pickle 
import time
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm 
 
from torchvision import models                                                                     
from convcap import convcap
from vggfeats import Vgg16Feats
from coco_loader import Scale
from PIL import Image
from test_beam import repeat_img
from beamsearch import beamsearch 

parser = argparse.ArgumentParser(description='PyTorch Convolutional Image \
    Captioning Model -- Caption Me')

parser.add_argument('model_dir', help='output directory to save models & results')
parser.add_argument('image_dir', help='directory containing input images \
                    supported formats .png, .jpg, .jpeg, .JPG')

parser.add_argument('-g', '--gpu', type=int, default=0,\
                    help='gpu device id')

parser.add_argument('--beam_size', type=int, default=1, \
                    help='beam size to use to generate captions') 

parser.add_argument('--attention', dest='attention', action='store_true', \
                    help='set caption model with attention in use (by default set)')

parser.add_argument('--no-attention', dest='attention', action='store_false', \
                    help='set caption model without attention in use')

parser.set_defaults(attention=True)

args = parser.parse_args()

def load_images(image_dir):
  """Load images from image_dir"""

  exts = ['.jpg', '.jpeg', '.png']
  imgs = torch.FloatTensor(torch.zeros(0, 3, 224, 224))
  imgs_fn = []

  img_transforms = transforms.Compose([
      Scale([224, 224]),
      transforms.ToTensor(),
      transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
        std = [ 0.229, 0.224, 0.225 ])
    ])

  for fn in os.listdir(image_dir):
    if(osp.splitext(fn)[-1].lower() in exts):
      imgs_fn.append(os.path.join(image_dir, fn))
      img = Image.open(os.path.join(image_dir, fn)).convert('RGB')
      img = img_transforms(img)
      imgs = torch.cat([imgs, img.unsqueeze(0)], 0)

  return imgs, imgs_fn

def captionme(args, modelfn):
  """Caption images in args.image_dir using checkpoint modelfn"""

  imgs, imgs_fn = load_images(args.image_dir)

  #For trained model released with the code
  batchsize = 1
  max_tokens = 15
  num_layers = 3 
  worddict_tmp = pickle.load(open('data/wordlist.p', 'rb'))
  wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
  wordlist = ['EOS'] + sorted(wordlist)
  numwords = len(wordlist)

  model_imgcnn = Vgg16Feats()
  model_imgcnn.cuda() 

  model_convcap = convcap(numwords, num_layers, is_attention = args.attention)
  model_convcap.cuda()

  print('[DEBUG] Loading checkpoint %s' % modelfn)
  checkpoint = torch.load(modelfn)
  model_convcap.load_state_dict(checkpoint['state_dict'])
  model_imgcnn.load_state_dict(checkpoint['img_state_dict'])

  model_imgcnn.train(False) 
  model_convcap.train(False)

  pred_captions = []
  for batch_idx, (img_fn) in \
    tqdm(enumerate(imgs_fn), total=len(imgs_fn)):
    
    img = imgs[batch_idx, ...].view(batchsize, 3, 224, 224)

    img_v = Variable(img.cuda())
    imgfeats, imgfc7 = model_imgcnn(img_v)

    b, f_dim, f_h, f_w = imgfeats.size()
    imgfeats = imgfeats.unsqueeze(1).expand(\
      b, args.beam_size, f_dim, f_h, f_w)
    imgfeats = imgfeats.contiguous().view(\
      b*args.beam_size, f_dim, f_h, f_w)

    b, f_dim = imgfc7.size()
    imgfc7 = imgfc7.unsqueeze(1).expand(\
      b, args.beam_size, f_dim)
    imgfc7 = imgfc7.contiguous().view(\
      b*args.beam_size, f_dim)

    beam_searcher = beamsearch(args.beam_size, batchsize, max_tokens)
  
    wordclass_feed = np.zeros((args.beam_size*batchsize, max_tokens), dtype='int64')
    wordclass_feed[:,0] = wordlist.index('<S>') 
    outcaps = np.empty((batchsize, 0)).tolist()

    for j in range(max_tokens-1):
      wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

      wordact, attn = model_convcap(imgfeats, imgfc7, wordclass)
      wordact = wordact[:,:,:-1]
      wordact_j = wordact[..., j]

      beam_indices, wordclass_indices = beam_searcher.expand_beam(wordact_j)  

      if len(beam_indices) == 0 or j == (max_tokens-2): # Beam search is over.
        generated_captions = beam_searcher.get_results()
        for k in range(batchsize):
            g = generated_captions[:, k]
            outcaps[k] = [wordlist[x] for x in g]
      else:
        wordclass_feed = wordclass_feed[beam_indices]
        imgfc7 = imgfc7.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
        imgfeats = imgfeats.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
        for i, wordclass_idx in enumerate(wordclass_indices):
          wordclass_feed[i, j+1] = wordclass_idx

    for j in range(batchsize):
      num_words = len(outcaps[j]) 
      if 'EOS' in outcaps[j]:
        num_words = outcaps[j].index('EOS')
      outcap = ' '.join(outcaps[j][:num_words])
      pred_captions.append({'img_fn': img_fn, 'caption': outcap})

  return pred_captions

def main():
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
  if(osp.exists(bestmodelfn)):
    pred_captions = captionme(args, bestmodelfn)
    resfile = osp.join(args.image_dir, 'captions.txt')
    with open(resfile, 'w') as fp:
      for item in pred_captions:
        fp.write('image: %s, caption: %s\n' % (item['img_fn'], item['caption']))
    print('[DEBUG] Captions written to file %s' % resfile)
  else:
    raise Exception('No checkpoint found %s' % bestmodelfn)

if __name__ == '__main__': 
  main()

