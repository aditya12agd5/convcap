import argparse
import numpy as np 
import time 
import pickle 
import itertools
import os.path as osp


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

from beamsearch import beamsearch 
from coco_loader import coco_loader
from torchvision import models                                                                    
from convcap import convcap
from vggfeats import Vgg16Feats
from evaluate import language_eval

def repeat_img(args, img_emb):
  b = img_emb.size(0)
  assert img_emb.dim() == 2

  img_emb_new = Variable(img_emb.data.new(img_emb.size(0)*args.beam_size, img_emb.size(1))).cuda()
  for k in range(b):
    start_idx = k * args.beam_size
    img_emb_new[start_idx:start_idx+args.beam_size, :] = img_emb[k, :].repeat(args.beam_size, 1)
  return img_emb_new

def test_beam(args, split, modelfn=None): 
  """Sample generation with beam-search"""

  t_start = time.time()
  data = coco_loader(args.coco_root, split=split, ncap_per_img=1)
  print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

  data_loader = DataLoader(dataset=data, num_workers=args.nthreads,\
    batch_size=args.batchsize, shuffle=False, drop_last=True)

  batchsize = args.batchsize
  max_tokens = data.max_tokens
  num_batches = np.int_(np.floor((len(data.ids)*1.)/batchsize))
  print('[DEBUG] Running test (w/ beam search) on %d batches' % num_batches)
  
  model_imgcnn = Vgg16Feats()
  model_imgcnn.cuda() 

  model_convcap = convcap(data.numwords, args.num_layers, is_attention=args.attention)
  model_convcap.cuda()

  print('[DEBUG] Loading checkpoint %s' % modelfn)
  checkpoint = torch.load(modelfn)
  model_convcap.load_state_dict(checkpoint['state_dict'])
  model_imgcnn.load_state_dict(checkpoint['img_state_dict'])

  model_imgcnn.train(False) 
  model_convcap.train(False)

  pred_captions = []
  for batch_idx, (imgs, _, _, _, img_ids) in \
    tqdm(enumerate(data_loader), total=num_batches):
    
    imgs = imgs.view(batchsize, 3, 224, 224)

    imgs_v = Variable(imgs.cuda())
    imgsfeats, imgsfc7 = model_imgcnn(imgs_v)

    b, f_dim, f_h, f_w = imgsfeats.size()
    imgsfeats = imgsfeats.unsqueeze(1).expand(\
      b, args.beam_size, f_dim, f_h, f_w)
    imgsfeats = imgsfeats.contiguous().view(\
      b*args.beam_size, f_dim, f_h, f_w)

    beam_searcher = beamsearch(args.beam_size, batchsize, max_tokens)
  
    wordclass_feed = np.zeros((args.beam_size*batchsize, max_tokens), dtype='int64')
    wordclass_feed[:,0] = data.wordlist.index('<S>') 
    imgsfc7 = repeat_img(args, imgsfc7)
    outcaps = np.empty((batchsize, 0)).tolist()

    for j in range(max_tokens-1):
      wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

      wordact, _ = model_convcap(imgsfeats, imgsfc7, wordclass)
      wordact = wordact[:,:,:-1]
      wordact_j = wordact[..., j]

      beam_indices, wordclass_indices = beam_searcher.expand_beam(wordact_j)  

      if len(beam_indices) == 0 or j == (max_tokens-2): # Beam search is over.
        generated_captions = beam_searcher.get_results()
        for k in range(batchsize):
            g = generated_captions[:, k]
            outcaps[k] = [data.wordlist[x] for x in g]
      else:
        wordclass_feed = wordclass_feed[beam_indices]
        imgsfc7 = imgsfc7.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
        imgsfeats = imgsfeats.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
        for i, wordclass_idx in enumerate(wordclass_indices):
          wordclass_feed[i, j+1] = wordclass_idx

    for j in range(batchsize):
      num_words = len(outcaps[j]) 
      if 'EOS' in outcaps[j]:
        num_words = outcaps[j].index('EOS')
      outcap = ' '.join(outcaps[j][:num_words])
      pred_captions.append({'image_id': img_ids[j], 'caption': outcap})

  scores = language_eval(pred_captions, args.model_dir, split)

  model_imgcnn.train(True) 
  model_convcap.train(True)

  return scores
