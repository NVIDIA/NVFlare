# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import numpy as np
import torch
from scipy.optimize import minimize

from .utils import move_ckpt


def get_encoder_keys(all_keys):
    """
    Get keys of encoder parameters
    """
    return list(filter(lambda x: 'backbone' in x, all_keys))


def get_decoder_keys(all_keys):
    """
    Get keys of decoder parameters
    """
    return list(filter(lambda x: 'decoders' in x, all_keys))


def get_model_soup(param_dict_list):
    """
    Get the average of parameters in list
    """
    soup_param_dict = {}
    layers = param_dict_list[0].keys()
    for layer in layers:
        soup_param_dict[layer] = torch.mean(
            torch.stack([param_dict_list[i][layer] for i in range(len(param_dict_list))]),
            dim=0,
        )
    return soup_param_dict


def get_delta_dict_list(param_dict_list, last_param_dict_list):
    """
    Get the difference between current and last parameters
    """
    # a list of length N, each element is a dict of delta parameters
    delta_dict_list = []
    layers = param_dict_list[0].keys()
    for i in range(len(param_dict_list)):
        delta_dict_list.append({})
        for layer in layers:
            delta_dict_list[i][layer] = param_dict_list[i][layer] - last_param_dict_list[i][layer]
    return delta_dict_list


def get_encoder_params(all_clients, ckpt):
    """
    Get encoder parameters from checkpoint
    """
    # encoder_param_list: a list of length n_st, each element is a dict of encoder parameters
    all_name_keys = [name for name, _ in all_clients[0]['model'].named_parameters()]
    encoder_keys = get_encoder_keys(all_name_keys)
    encoder_param_dict_list = []
    layers = []
    shapes = []

    for model_idx in range(len(ckpt)):
        param_dict = {}
        for key in encoder_keys:
            # key=prefix+'.'+layer
            prefix, layer = key.split('.', 1)
            param_dict[layer] = ckpt[model_idx][key]
        encoder_param_dict_list.append(param_dict)

    # Get layers and shapes (same for all encoders)
    for key in encoder_keys:
        layers.append(key.split('.', 1)[1])
        shapes.append(ckpt[0][key].shape)

    return encoder_param_dict_list, encoder_keys, layers, shapes


def get_decoder_params(all_clients, ckpt):
    """
    Get decoder parameters from checkpoint
    """
    N = len(all_clients)
    n_st = sum([len(all_clients[i]['tasks']) == 1 for i in range(N)])  # number of st clients
    K = sum([len(all_clients[i]['tasks']) for i in range(N)])  # number of decoders

    decoder_keys = []
    layers = []
    shapes = []

    for idx in range(N):
        all_name_keys = [key for key, _ in all_clients[idx]['model'].named_parameters()]
        decoder_keys += get_decoder_keys(all_name_keys)
    decoder_keys = list(set(decoder_keys))

    decoder_param_dict_list = []
    decoders_prefix = []
    # st client decoders
    for model_idx in range(n_st):
        assert len(all_clients[model_idx]['tasks']) == 1
        param_dict = {}
        for key in decoder_keys:
            if key in ckpt[model_idx].keys():
                # key=prefix+'.'+layer
                prefix = key.split('.', 2)[0] + '.' + key.split('.', 2)[1]  # decoders.task
                layer = key.split('.', 2)[2]
                param_dict[layer] = ckpt[model_idx][key]

                if model_idx == 0:
                    layers.append(layer)
                    shapes.append(ckpt[0][key].shape)

        decoders_prefix.append(prefix)
        decoder_param_dict_list.append(param_dict)

    # mt client decoders
    for model_idx in range(n_st, N):
        prefix_list = []  # decoder prefixs in one mt client
        for task in all_clients[model_idx]['tasks']:
            prefix_list.append('decoders.' + task)
        prefix_list = sorted((prefix_list))  # keep the order

        for i, prefix in enumerate(prefix_list):
            # Get each task-specific decoder
            param_dict = {}
            for key in decoder_keys:
                if key in ckpt[model_idx].keys() and prefix in key:
                    layer = key.split('.', 2)[2]
                    param_dict[layer] = ckpt[model_idx][key]

                    if model_idx == 0 and i == 0:
                        layers.append(layer)
                        shapes.append(ckpt[0][key].shape)

            decoder_param_dict_list.append(param_dict)
        decoders_prefix += prefix_list

    assert len(decoders_prefix) == K
    assert len(decoder_param_dict_list) == K

    return decoder_param_dict_list, decoders_prefix, decoder_keys, layers, shapes


def get_ca_delta(flatten_delta_list, alpha, rescale=1):
    """
    Solve for aggregated conflict-averse delta
    """
    N = len(flatten_delta_list)
    grads = torch.stack(flatten_delta_list).t()  # [d , N]
    GG = grads.t().mm(grads).cpu()  # [N, N]
    g0_norm = (GG.mean() + 1e-8).sqrt()

    x_start = np.ones(N) / N
    bnds = tuple((0, 1) for x in x_start)
    cons = {'type': 'eq', 'fun': lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
            + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    ww = torch.Tensor(res.x).to(grads.device)
    gw = (grads * ww.reshape(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        final_update = g
    elif rescale == 1:
        final_update = g / (1 + alpha**2)
    else:
        final_update = g / (1 + alpha)

    return final_update


def flatten_param(param_dict_list, layers):
    """
    Flatten a dict of parameters into a vector
    """
    flatten_list = [
        torch.cat([param_dict_list[idx][layer].flatten() for layer in layers]) for idx in range(len(param_dict_list))
    ]
    assert len(flatten_list[0].shape) == 1
    return flatten_list


def unflatten_param(flatten_list, shapes, layers):
    """
    Unflatten a vector into a dict of parameters
    """
    param_dict_list = []
    for model_idx in range(len(flatten_list)):
        start = 0
        param_dict_list.append({})
        for layer, shape in zip(layers, shapes):
            end = start + np.prod(shape)
            param_dict_list[model_idx][layer] = flatten_list[model_idx][start:end].reshape(shape)
            start = end
    return param_dict_list


def aggregate(
    all_clients,
    save_ckpt,
    last_ckpt,
    hyperweight=None,
    encoder_agg='none',
    decoder_agg='none',
    ca_c=0.4,
):
    '''
    Main aggregation function after assigned epochs of local training
    '''
    assert len(all_clients) == len(save_ckpt)
    N = len(all_clients)
    n_st = sum([len(client['tasks']) == 1 for client in all_clients])
    n_mt_tasks = [len(all_clients[i]['tasks']) for i in range(n_st, N)]  # number of tasks for each mt client

    if encoder_agg == 'none' and decoder_agg == 'none':
        return  # no aggregation

    update_ckpt = copy.deepcopy(save_ckpt)  # store updated parameters

    # Get encoder parameter list
    encoder_param_list, encoder_keys, enc_layers, enc_shapes = get_encoder_params(all_clients, save_ckpt)

    # Encoder agg
    if encoder_agg == 'none':
        del encoder_param_list
        pass

    elif encoder_agg in ['fedavg']:
        new_encoder_param = get_model_soup(encoder_param_list)

        for model_idx in range(N):
            for key in encoder_keys:
                layer = key.split('.', 1)[1]
                update_ckpt[model_idx][key] = new_encoder_param[layer]

        del encoder_param_list, new_encoder_param

    elif encoder_agg in ['conflict_averse']:
        last_encoder_param_list, _, _, _ = get_encoder_params(all_clients, last_ckpt)
        encoder_delta_list = get_delta_dict_list(encoder_param_list, last_encoder_param_list)

        # Flatten
        flatten_last_encoder = flatten_param(last_encoder_param_list, enc_layers)
        del last_encoder_param_list
        flatten_encoder_delta = flatten_param(encoder_delta_list, enc_layers)
        del encoder_delta_list

        # Solve for aggregated conflict-averse delta
        flatten_delta_update = get_ca_delta(flatten_encoder_delta, ca_c)  # flattened tensor

        # Homo aggregation
        group = [0]
        homo_avg = flatten_encoder_delta[0]
        i = 1
        while i < N:
            if (
                all_clients[i]['dataname'] == all_clients[i - 1]['dataname']
                and all_clients[i]['tasks'] == all_clients[i - 1]['tasks']
            ):
                homo_avg += flatten_encoder_delta[i]
                group.append(i)
            else:
                homo_avg /= len(group)
                for j in group:
                    flatten_encoder_delta[j] = homo_avg
                group = [i]
                homo_avg = flatten_encoder_delta[i]
            i += 1
        homo_avg /= len(group)
        for j in group:
            flatten_encoder_delta[j] = homo_avg

        # Personalized update
        assert hyperweight['enc'] is not None
        flatten_new_encoder = hyperweight['enc'](flatten_last_encoder, flatten_encoder_delta, flatten_delta_update)
        # Record output of hyperweight for backprop
        hyperweight['last_enc_output'] = flatten_new_encoder

        del flatten_last_encoder, flatten_encoder_delta, flatten_delta_update

        new_encoder_param_list = unflatten_param(flatten_new_encoder, enc_shapes, enc_layers)

        for model_idx in range(N):
            for key in encoder_keys:
                layer = key.split('.', 1)[1]
                update_ckpt[model_idx][key] = new_encoder_param_list[model_idx][layer]

        del new_encoder_param_list

    else:
        raise NotImplementedError

    # Get decoder parameter list and prefix
    decoder_param_list, decoders_prefix, decoder_keys, dec_layers, dec_shapes = get_decoder_params(
        all_clients, save_ckpt
    )

    # Decoder agg
    if decoder_agg == 'none':
        del decoder_param_list
        pass

    elif decoder_agg in ['fedavg']:
        new_decoder_param = get_model_soup(decoder_param_list)

        for i, prefix in enumerate(decoders_prefix):
            # first st clients then mt clients
            if i >= n_st:
                model_idx = n_st + (i - n_st) // (n_mt_tasks[0])
            else:
                model_idx = i

            for layer in dec_layers:
                update_ckpt[model_idx][prefix + '.' + layer] = new_decoder_param[layer]

        del decoder_param_list, new_decoder_param

    elif decoder_agg in ['cross_attention']:
        assert hyperweight['dec'] is not None
        last_decoder_param_list, _, _, _, _ = get_decoder_params(all_clients, last_ckpt)
        decoder_delta_list = get_delta_dict_list(decoder_param_list, last_decoder_param_list)

        # Personalized update
        new_decoder_param_list = hyperweight['dec'](last_decoder_param_list, decoder_delta_list)
        # Record output of hyperweight for backprop
        hyperweight['last_dec_output'] = new_decoder_param_list

        for i, (prefix, new_decoder_param) in enumerate(zip(decoders_prefix, new_decoder_param_list)):
            # first st clients then mt clients
            if i >= n_st:
                tmp = i - n_st
                k = 0
                while tmp >= n_mt_tasks[k]:
                    tmp -= n_mt_tasks[k]
                    k += 1
                model_idx = n_st + k
            else:
                model_idx = i

            for layer in new_decoder_param.keys():
                update_ckpt[model_idx][prefix + '.' + layer] = new_decoder_param[layer]

        del last_decoder_param_list, decoder_delta_list

    else:
        raise NotImplementedError

    # Update all models
    update_ckpt = move_ckpt(update_ckpt, 'cuda')
    for model_idx in range(N):
        all_clients[model_idx]['model'].load_state_dict(update_ckpt[model_idx])

    del update_ckpt


def update_hyperweight(all_nets, hyperweight, save_ckpt, last_ckpt):
    '''
    Update hyperweights with corresponding delta of encoder and decoder parameters
    '''
    if 'enc' in hyperweight.keys():
        # Get encoder parameter list and prefix
        encoder_param_list, encoder_keys, enc_layers, enc_shapes = get_encoder_params(all_nets, save_ckpt)
        last_encoder_param_list, _, _, _ = get_encoder_params(all_nets, last_ckpt)

        # Calculate difference between current and last encoder parameters
        diff_list = get_delta_dict_list(last_encoder_param_list, encoder_param_list)
        flatten_diff = flatten_param(diff_list, enc_layers)

        # Update hyperweight
        hyperweight['enc'].train()
        optimizer = hyperweight['enc_optimizer']
        optimizer.zero_grad()

        torch.autograd.backward(hyperweight['last_enc_output'], flatten_diff, retain_graph=True)

        optimizer.step()

    if 'dec' in hyperweight.keys():
        # Get decoder parameter list and prefix
        decoder_param_list, decoders_prefix, decoder_keys, dec_layers, dec_shapes = get_decoder_params(
            all_nets, save_ckpt
        )
        last_decoder_param_list, last_decoders_prefix, _, _, _ = get_decoder_params(all_nets, last_ckpt)
        assert decoders_prefix == last_decoders_prefix

        # Calculate difference between current and last decoder parameters
        diff_list = get_delta_dict_list(last_decoder_param_list, decoder_param_list)

        # Update hyperweight
        hyperweight['dec'].train()
        optimizer = hyperweight['dec_optimizer']
        optimizer.zero_grad()

        for i in range(len(decoder_param_list)):
            # Construct dict of parameters into list
            last_output = list(map(lambda x: hyperweight['last_dec_output'][i][x], dec_layers))
            diff_param = list(map(lambda x: diff_list[i][x], dec_layers))
            torch.autograd.backward(last_output, diff_param, retain_graph=True)

        optimizer.step()
