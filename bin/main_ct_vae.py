#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vganapa1, vdumont
"""

import sys
sys.path.append('/Users/vganapa1/Dropbox/Github/CT_VAE')

from ctvae.main_ct_vae import get_args, CT_VAE

if __name__ == "__main__":
    args = get_args()
    CT_VAE(**vars(args))
