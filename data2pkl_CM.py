# -*- coding:utf-8 -*-
import os
import sys
import pickle
import numpy as np

def pkl_save(pkl_data, save_dir):
    fp = open(save_dir, 'wb')
    pickle.dump(pkl_data, fp)
    fp.close()
    return print('Data saved in file: ' + save_dir)

def pssm_load(pssm_dir, start=2, end=22, tail='.pssm'):
    fp = open(pssm_dir + tail, 'r')
    lines = fp.readlines()
    fp.close()
    pssm = []
    for line in lines:
        temp = line.split()
        if len(temp) == 0:
            continue  
        pssm.append(temp[start:end])
    return np.array(pssm, dtype=np.float32)

def pai_load(pai_dir, tail='.pai'):
    fp = open(pai_dir + tail, 'r')
    lines = fp.readlines()
    fp.close()
    map_shape = int(lines[-1].split()[1])
    mcp = np.zeros([map_shape, map_shape], dtype=np.float16)
    mi = np.zeros([map_shape, map_shape], dtype=np.float16)
    nmi = np.zeros([map_shape, map_shape], dtype=np.float16)
    for line in lines:
        temp = line.split()
        mcp[int(temp[0]) - 1, int(temp[1]) - 1] = float(temp[2])
        mi[int(temp[0]) - 1, int(temp[1]) - 1] = float(temp[3])
        nmi[int(temp[0]) - 1, int(temp[1]) - 1] = float(temp[4])
    mcp = mcp + mcp.transpose()
    mi = mi + mi.transpose()
    nmi = nmi + nmi.transpose()
    return mcp, mi, nmi

def fasta_load(fasta_dir, tail='.fasta'):
    fp = open(fasta_dir + tail, 'r')
    lines = fp.readlines()
    fp.close()
    sequence = ''
    for line in lines[1:]:
        sequence = sequence + line.split()[0]
    return sequence

def norm_wt_diag(matrix):
    mat_len = np.shape(matrix)[0]
    mat_sum = np.sum(matrix)
    mat_mean = mat_sum/mat_len/(mat_len-1)
    mat_std = np.sqrt(np.sum(np.square(matrix-mat_mean))/mat_len/(mat_len-1))
    return np.where(np.eye(mat_len), 0, (matrix-mat_mean)/mat_std)

def files2pkl(id_list, pkl_dir, pkl_name, fasta_dir, pssm_dir, ccm_dir, pai_dir):
    pkl = []
    report_i = 0
    for pdbid in id_list:
        report_i = report_i + 1
        sequence = fasta_load(fasta_dir+pdbid)
        pssm_ss3_acc = pssm_load(pssm_dir+pdbid, start=0, end=46, tail='.feat')
        pssm = pssm_ss3_acc[:, 0:20]
        acc = pssm_ss3_acc[:, 40:43]
        ss3 = pssm_ss3_acc[:, 43:46]
        ccmpredZ = np.loadtxt(ccm_dir+pdbid+'.mat', dtype=np.float32)
        ccmpredZ = norm_wt_diag(ccmpredZ)
        seq_length = np.shape(ccmpredZ)[0]
        ccm_mask = np.zeros([seq_length, seq_length])
        for i in range(5):
            ccm_mask = np.eye(seq_length, k=i+1) + ccm_mask
        ccm_mask = ccm_mask + ccm_mask.transpose() + np.eye(seq_length)
        ccmpredZ = np.where(ccm_mask, 0, ccmpredZ).astype(dtype=np.float16)
        mcp, mi, nmi = pai_load(pai_dir+pdbid)
        mcp = norm_wt_diag(mcp)
        mi = norm_wt_diag(mi)
        OtherPairs = np.transpose(np.array([mi, mcp, nmi]), axes=[2, 1, 0])
        contactMatrix = np.ones([seq_length, seq_length], dtype=np.int8)
        pkl_dict = {b'name': pdbid.encode('utf-8'),
                    b'sequence': sequence.encode('utf-8'),
                    b'PSSM': pssm,
                    b'SS3': ss3,
                    b'ACC': acc,
                    b'ccmpredZ': ccmpredZ,
                    b'OtherPairs': OtherPairs,
                    b'contactMatrix': contactMatrix}
        pkl.append(pkl_dict)
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)
    return pkl_save(pkl, pkl_dir + '/' + pkl_name)

temp = sys.argv[1]

id_list = temp.split()
files2pkl(id_list, './example/', sys.argv[1]+'_CM.pkl', './example/', './example/', './example/', './example/')
