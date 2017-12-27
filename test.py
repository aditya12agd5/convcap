import os
import os.path as osp
import argparse
import numpy as np 
import json
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
 
from coco_loader import coco_loader
from torchvision import models                                                                     
from convcap import convcap
from vggfeats import Vgg16Feats
from evaluate import language_eval


def save_test_json(preds, resFile):
  print('Writing %d predictions' % (len(preds)))
  json.dump(preds, open(resFile, 'w')) 

def test(args, split, modelfn=None, model_convcap=None, model_imgcnn=None):
  """Runs test on split=val/test with checkpoint file modelfn or loaded model_*"""

  t_start = time.time()
  data = coco_loader(args.coco_root, split=split, ncap_per_img=1)
  print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

  data_loader = DataLoader(dataset=data, num_workers=args.nthreads,\
    batch_size=args.batchsize, shuffle=False, drop_last=True)

  batchsize = args.batchsize
  max_tokens = data.max_tokens
  num_batches = np.int_(np.floor((len(data.ids)*1.)/batchsize))
  print('[DEBUG] Running inference on %s with %d batches' % (split, num_batches))

  if(modelfn is not None):
    model_imgcnn = Vgg16Feats()
    model_imgcnn.cuda() 

    model_convcap = convcap(data.numwords, args.num_layers, is_attention=args.attention)
    model_convcap.cuda()

    print('[DEBUG] Loading checkpoint %s' % modelfn)
    checkpoint = torch.load(modelfn)
    model_convcap.load_state_dict(checkpoint['state_dict'])
    model_imgcnn.load_state_dict(checkpoint['img_state_dict'])
  else:
    model_imgcnn = model_imgcnn
    model_convcap = model_convcap

  model_imgcnn.train(False) 
  model_convcap.train(False)

  pred_captions = []
  #Test epoch
  for batch_idx, (imgs, _, _, _, img_ids) in \
    tqdm(enumerate(data_loader), total=num_batches):
    
    imgs = imgs.view(batchsize, 3, 224, 224)

    imgs_v = Variable(imgs.cuda())
    imgsfeats, imgsfc7 = model_imgcnn(imgs_v)
    _, featdim, feat_h, feat_w = imgsfeats.size()
  
    wordclass_feed = np.zeros((batchsize, max_tokens), dtype='int64')
    wordclass_feed[:,0] = data.wordlist.index('<S>') 

    outcaps = np.empty((batchsize, 0)).tolist()

    for j in range(max_tokens-1):
      wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

      wordact, _ = model_convcap(imgsfeats, imgsfc7, wordclass)

      wordact = wordact[:,:,:-1]
      wordact_t = wordact.permute(0, 2, 1).contiguous().view(batchsize*(max_tokens-1), -1)

      wordprobs = F.softmax(wordact_t).cpu().data.numpy()
      wordids = np.argmax(wordprobs, axis=1)

      for k in range(batchsize):
        word = data.wordlist[wordids[j+k*(max_tokens-1)]]
        outcaps[k].append(word)
        if(j < max_tokens-1):
          wordclass_feed[k, j+1] = wordids[j+k*(max_tokens-1)]

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
 
