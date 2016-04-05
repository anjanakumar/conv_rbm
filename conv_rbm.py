#Convolutional RBM
#Developed by jpvmm
#FutureBox.AI


from __future__ import division
import sys
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.shared_randomstreams import RandomStreams

from make_patches import make_patch
import extend
import glob
import pickle


class CRBM:
    ''' A Convolutional RBM '''
    def __init__(self, lr,
                 patch_size, bases,
                 t_sparsity, W = None,
                 vbias = None, hbias = None,
                 step = None):

        self.patch_size = patch_size
        self.bases = bases #Bases are like feature maps in conv_nets

        self.visibles = (patch_size,patch_size)
        self.hidden = 1
        self.lr = lr
        self.t_sparsity = t_sparsity

        generator = RandomStreams(seed = 234)


        if step is None:
            self.step = 0
        
        if W is None:
            initial_W = np.asarray(
                np.random.randn(bases, patch_size, patch_size),
                dtype = theano.config.floatX)

            W = theano.shared(value = initial_W, name = 'W', borrow = True)
        
        self.W = W

        if vbias is None:
            vbias = theano.shared(value = np.ones(self.visibles, dtype = theano.config.floatX),
            name = 'vbias',
            borrow = True )
        
        if hbias is None:
            hbias = theano.shared(value = np.ones(self.hidden ** self.t_sparsity, dtype = theano.config.floatX),
            name = 'hbias',
            borrow = True)
        
        self.hbias = hbias
        self.vbias = vbias
      


if __name__ ==  '__main__':

    c1 = CRBM(lr = 0.1, patch_size = 2, bases = 2, t_sparsity = -1)






