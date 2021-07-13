# -*- coding:utf-8 -*-
import pickle
import numpy as np

def pkl_read(pkl_file, encoding='bytes'):
    print('Loading from: ' + pkl_file)
    fp = open(pkl_file, 'rb')
    contents = pickle.load(fp, encoding=encoding)
    fp.close()
    return contents

def pkl_save(pkl_data, save_dir):
    fp = open(save_dir, 'wb')
    pickle.dump(pkl_data, fp)
    fp.close()
    return print('Data saved in file: ' + save_dir)

def sort_len(seq_info):
    length_subs = []
    for i, data in enumerate(seq_info):
        length = len(data[b'sequence'])
        length_sub = [length, i, data[b'name'].decode('utf-8')]
        length_subs.append(length_sub)
    return sorted(length_subs, key=lambda seq_length: seq_length[0], reverse=True)

def sub2train_fl(length_subs, seq_infos, out_length=700):
    max_length = 0
    seq_ins = []
    pairwize_infos = []
    maps = []
    for length, sub, _ in length_subs:
        if length > max_length:
            max_length = length
        pad_length = out_length - length
        seq_data = seq_infos[sub]
        pssm = seq_data[b'PSSM']
        ss3 = seq_data[b'SS3']
        acc = seq_data[b'ACC']
        sequence = seq_data[b'sequence']
        sequence = str(sequence, encoding = "utf-8")  
        seq_in = np.concatenate([pssm, ss3, acc], axis=1)
        seq_ins.append(np.pad(seq_in, ((0, pad_length), (0, 0)), 'constant'))
    return sequence, np.array(seq_ins)[:, :, :]

def sub2train_fl_CM(length_subs, seq_infos, out_length=700):
    max_length = 0
    seq_ins = []
    pairwize_infos = []
    maps = []
    for length, sub, _ in length_subs:
        if length > max_length:
            max_length = length
        pad_length = out_length - length
        seq_data = seq_infos[sub]
        pssm = seq_data[b'PSSM']
        ss3 = seq_data[b'SS3']
        acc = seq_data[b'ACC']
        lipidcontact = seq_data[b'lipidcontact'][:, :1]
        lipidcontact2 = seq_data[b'lipidcontact2'][:, :1]
        seq_in = np.concatenate([pssm, ss3, acc, lipidcontact, lipidcontact2], axis=1)
        seq_ins.append(np.pad(seq_in, ((0, pad_length), (0, 0)), 'constant'))
        ccmpredz = seq_data[b'ccmpredZ'][:, :, np.newaxis]
        otherpairs = seq_data[b'OtherPairs']
        pairwize_info = np.concatenate([ccmpredz, otherpairs[:, :, :-1]], axis=2)
        pairwize_infos.append(np.pad(pairwize_info, ((0, pad_length), (0, pad_length), (0, 0)), 'constant'))
        maps.append(np.pad(seq_data[b'contactMatrix'], ((0, pad_length), (0, pad_length)),
                           'constant', constant_values=(0, -1)))
    mask = np.pad(np.ones(max_length, dtype=bool), (0, out_length - max_length), 'constant')
    return np.array(seq_ins)[:, np.newaxis, :, :], np.array(pairwize_infos), np.array(maps)[:, :, :, np.newaxis], mask
