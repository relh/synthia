#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import time
from datetime import date

import drms
import numpy as np
import torch
from astropy.io import fits

from pad import pad_to, unpad
from unet import UNet


def make_output(date, all_nets, all_params, tt, targets, outdir):
    print('check if exists already')
    exists = 0
    for target in targets: 
        outName = "synode.hmi2hinode.%s.%s" % (date, target)
        os.makedirs(outdir + tt + '/' + target + '/', exist_ok=True)
        if os.path.exists(outdir + tt + '/' + target + '/%s.fits' % (outName)):
            exists += 1
    if exists == len(targets): return

    iquvs = ['continuum'] + sorted([y + str(x) for x in range(6) for y in ['I', 'Q', 'U', 'V']])
    print('make drms client')
    client = drms.Client()
    results_c, filenames_c = client.query(f'hmi.Ic_720s[{date}]', key=drms.const.all, seg=iquvs)
    results, filenames = client.query(f'hmi.s_720s[{date}]', key=drms.const.all, seg=iquvs)

    try:
        print('fetch JSOC data')
        input_tensor = torch.zeros((len(iquvs), 4096, 4096)).float()
        for i, channel in enumerate(iquvs):
            this_filenames = filenames_c if i == 0 else filenames
            fname = f'http://jsoc.stanford.edu{this_filenames[channel][0]}'
            print(fname)
            original_fits_data = fits.getdata(fname).astype(np.float32)
            fits_data = np.nan_to_num(original_fits_data)
            input_tensor[i] = torch.tensor(fits_data)
    except Exception as e:
        print(str(e) + str(date) + ' broken')
        return

    for iii, target in enumerate(targets):
        start = time.time()
        print('---' + target + '---')
        params = all_params[target]
        net = all_nets[target]

        print('normalize tensor')
        med = params['X_median']
        scale = np.atleast_1d(1.5 * (params['X_iqr'] + 1e-1))

        if 'br' in target or 'bp' in target or 'bt' in target:
            ymed = float(params['Y_median'])
            yscale = float(np.atleast_1d(1.5 * (params['Y_iqr'] + 1e-1)))
            new_input_tensor = ((input_tensor[1:] - med[:, np.newaxis, np.newaxis]) / scale[:, np.newaxis, np.newaxis]).float()
            my_iquvs = iquvs[1:]
        else:
            new_input_tensor = ((input_tensor[:] - med[:, np.newaxis, np.newaxis]) / scale[:, np.newaxis, np.newaxis]).float()
            my_iquvs = iquvs

        print('chop up tensor')
        ii = 0
        NUM_CHUNKS = 4
        CHUNK_SIZE = 1024
        FINE_CHUNKS = NUM_CHUNKS + (NUM_CHUNKS - 1)
        FINE_OFFSET = CHUNK_SIZE // 2
        cutup_input_tensor = torch.zeros((FINE_CHUNKS**2, len(my_iquvs), FINE_OFFSET, FINE_OFFSET))

        if target == 'b_fill_factor' or target == 'field': max_divisor = 5000.0
        else: max_divisor = 180.0

        print('run each panel through network')
        h_list = []
        v_list = []

        with torch.set_grad_enabled(False):
            for y in range(FINE_CHUNKS):
                h_list = []
                for x in range(FINE_CHUNKS):
                    sy = y * FINE_OFFSET
                    sx = x * FINE_OFFSET
                    ey = sy + CHUNK_SIZE
                    ex = sx + CHUNK_SIZE

                    im_inp = new_input_tensor[:, sy:ey, sx:ex].unsqueeze(0).to(iii)
                    im_inp, pads = pad_to(im_inp, 8)
                    pred = net(im_inp)
                    pred = unpad(pred, pads)

                    if target[0] == 'b' and target[1] != '_':
                        pred_im = ((pred * yscale) + ymed).float().squeeze() #.cpu()
                    else:
                        pred = torch.nn.functional.softmax(pred.squeeze(), dim=0)

                        # find the max probability bin
                        _, max_indices = torch.max(pred, 0)
                        max_indices = max_indices.unsqueeze(0)

                        # make an ordinal scatter against the one hot args.bins
                        max_mask = torch.zeros((80, pred.shape[1], pred.shape[2])).to(iii)
                        scatter_ones = torch.ones(max_indices.shape).to(iii)
                        scatter_range = torch.arange(80).unsqueeze(1).unsqueeze(1).float().to(iii)

                        up_max_indices = (max_indices + 1).clamp(0, 80 - 1)
                        down_max_indices = (max_indices - 1).clamp(0, 80 - 1)

                        mod_max_indices = max_mask.scatter_(0, max_indices, scatter_ones)
                        mod_max_indices = mod_max_indices.scatter_(0, up_max_indices, scatter_ones)
                        mod_max_indices = mod_max_indices.scatter_(0, down_max_indices, scatter_ones)

                        masked_probabilities = (mod_max_indices * pred)
                        normed_probabilities = masked_probabilities / masked_probabilities.sum(dim=0)
                        indices = (normed_probabilities * scatter_range).sum(dim=0)
                        pred_im = (indices.float() / ((80 - 1) / max_divisor)) #.cpu()

                    if y == 0:                 y_range = [0, 768]
                    elif y == FINE_CHUNKS - 1: y_range = [256, 1024]
                    else:                      y_range = [256, 768]

                    if x == 0:                 x_range = [0, 768]
                    elif x == FINE_CHUNKS - 1: x_range = [256, 1024]
                    else:                      x_range = [256, 768]

                    borderless_chunk = pred_im[y_range[0]:y_range[1], x_range[0]:x_range[1]]
                    h_list.append(borderless_chunk)
                h_list = torch.hstack(h_list)
                v_list.append(h_list)

        print('stitch back together')
        v_list = torch.vstack(v_list)

        print('save output')
        outName = "synode.hmi2hinode.%s.%s" % (date, target)
        v_list = np.asarray(v_list.cpu())
        v_list[original_fits_data != original_fits_data] = None
        hdu = fits.PrimaryHDU(v_list)
        hdu.writeto(outdir + tt + '/' + target + "/%s.fits" % (outName), overwrite=True)
        print(time.time() - start)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--timelist', default='./timelists/2015.11.timelist.txt', type=str, help='List of times to process')
    p.add_argument('--outdir', default='./output/', type=str, help='directory to save output to')
    args = p.parse_args()

    with open(args.timelist) as ff:
        dates = [x.strip() for x in ff.readlines()]
    dates = [x.replace('-', '.').replace(' ', '_') for x in dates]

    nets = {}
    params = {}
    targets = ['b_fill_factor', 'field']#, 'inclination', 'azimuth', 'br', 'bp', 'bt']):

    for iii, target in enumerate(targets):
        print('load normalization parameters')
        number = 17 if iii >= 4 else 9
        inputs = ['i', 'q', 'u', 'v'] if iii >= 4 else ['contin', 'i', 'q', 'u', 'v']
        params_path = './params/params_v{}_{}_{}_{}_{}.pkl'.format(str(number), 'tiles', str('-'.join(inputs)), str('-'.join([target])), str(80))
        print(params_path)
        with open(params_path, 'rb') as handle:
            params[target] = pickle.load(handle)

        print('load network data')
        nets[target] = UNet(24 if iii >= 4 else 25, 1, False, dropout=0.3, regression=(True if iii >= 4 else False), bins=80, bc=64).to(iii)
        load_dict = torch.load(os.path.join('./models/', target + '.pth'), map_location=f'cuda:{iii}')
        nets[target].load_state_dict(load_dict['model'])

    print('retry 5 times because DRMS disconnects frequently')
    for _ in range(5):
        for i, d in enumerate(dates):
            print(str(i) + ' ' + str(d))
            make_output(d, nets, params, args.timelist, targets, args.outdir)
