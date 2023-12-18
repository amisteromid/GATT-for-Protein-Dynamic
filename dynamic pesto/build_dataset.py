import re
import h5py
import numpy as np
import torch as pt
from glob import glob
from tqdm import tqdm
from collections import Counter

from utils.structure import StructuresDataset, concatenate_chains, clean_structure, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits, remove_duplicate_tagged_subunits, reorganize_structure
from utils.configs import config_dataset, config_encoding
from utils.data_encoding import encode_features, encode_structure, extract_topology, extract_dynamic_features, mean_coordinates



def locate_contacts(xyz_i, xyz_j, r_thr, device=pt.device("cpu")):
    with pt.no_grad():
        # send data to device
        if isinstance(xyz_i, pt.Tensor):
            X_i = xyz_i.to(device)
            X_j = xyz_j.to(device)
        else:
            X_i = pt.from_numpy(xyz_i).to(device)
            X_j = pt.from_numpy(xyz_j).to(device)

        # compute distance matrix between subunits
        D = pt.norm(X_i.unsqueeze(1) - X_j.unsqueeze(0), dim=2)

        # find contacts
        ids_i, ids_j = pt.where(D < r_thr)

        # get contacts distances
        d_ij = D[ids_i, ids_j]

    return ids_i.cpu(), ids_j.cpu()

def merge_contacts(tensor_list, num_conformers):
	# merge contacts of a pair over all conformers with their frequency
    unique_tensors = {}
    #print (tensor_list)
    for tensor in tensor_list:
        if tensor in unique_tensors:
            unique_tensors[tensor] += 1
        else:
            unique_tensors[tensor] = 1
    new_dic = {'ids': pt.stack([key for key in unique_tensors.keys()], dim=0), 'freq': [value/num_conformers for value in unique_tensors.values()]}

    return new_dic

def extract_all_contacts(subunits, r_thr, device=pt.device("cpu")):
    # get subunits names
    snames = list(subunits)

    # extract interfaces
    contacts_dict = {}
    for i in range(len(snames)):
        # current selection chain
        cid_i = snames[i]

        for j in range(i+1, len(snames)):
            # current selection chain
            cid_j = snames[j]
            
            contacts_tmp = []
            for conformer in subunits[cid_i].keys():
                # find contacts
                ids_i, ids_j = locate_contacts(subunits[cid_i][conformer]['xyz'], subunits[cid_j][conformer]['xyz'], r_thr, device=device)
                # now...
                if (ids_i.shape[0] > 0) and (ids_j.shape[0] > 0):
                    contacts_tmp.extend([pair for pair in pt.stack([ids_i,ids_j], dim=1)])
            num_conformers = len(subunits[cid_i])
            contacts_tmp = [tuple(t.tolist()) for t in contacts_tmp]
            contacts_tmp = Counter(contacts_tmp)
            ids = pt.stack([pt.tensor(i) for i in contacts_tmp.keys()],dim=0)
            freq = pt.tensor(list(contacts_tmp.values()))/num_conformers
            
            if f'{cid_i}' in contacts_dict:
                contacts_dict[f'{cid_i}'].update({f'{cid_j}': {'ids': ids, 'freq': freq}})
            else:
                contacts_dict[f'{cid_i}'] = {f'{cid_j}': {'ids': ids, 'freq': freq}}

            if f'{cid_j}' in contacts_dict:
                contacts_dict[f'{cid_j}'].update({f'{cid_i}': {'ids': ids[:,[1,0]], 'freq': freq}})
            else:
                contacts_dict[f'{cid_j}'] = {f'{cid_i}': {'ids': ids[:,[1,0]], 'freq': freq}}
                
            
    return contacts_dict

def contacts_types(s0, M0, s1, M1, ids, freq, molecule_ids, device=pt.device("cpu")):
    # molecule types for s0 and s1
    c0 = pt.from_numpy(s0['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)
    c1 = pt.from_numpy(s1['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)
    
    # filter by freq
    ids = ids[freq>=0.5]
    # categorize contacts (h.shape -> (len(molecule_ids),len(ids), len(ids)))
    H = (c1[ids[:,1]].unsqueeze(1) & c0[ids[:,0]].unsqueeze(2))

    # residue indices of contacts
    rids0 = pt.where(M0[ids[:,0]])[1]
    rids1 = pt.where(M1[ids[:,1]])[1]

    # create detailed contact map: automatically remove duplicated atom-atom to residue-residue contacts
    Y = pt.zeros((M0.shape[1], M1.shape[1], H.shape[1], H.shape[2]), device=device, dtype=pt.bool)
    Y[rids0, rids1] = H
    
    # define assembly type fingerprint matrix
    T = pt.any(pt.any(Y, dim=1), dim=0)
    return Y, T
    
def pack_contacts_data(Y, T):
    return {
        'Y':pt.stack(pt.where(Y), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        'Y_shape': Y.shape, 'ctype': T.cpu().numpy(),
    }


def pack_structure_data(X, qe, qr, qn, M, ids_topk, dccm, rmsf):
    return {
        'X': X.cpu().numpy().astype(np.float32),
        'ids_topk': ids_topk.cpu().numpy().astype(np.uint16),
        'qe':pt.stack(pt.where(qe > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qr':pt.stack(pt.where(qr > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qn':pt.stack(pt.where(qn > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'M':pt.stack(pt.where(M), dim=1).cpu().numpy().astype(np.uint16),
        'dccm':dccm.cpu().numpy().astype(np.float32),
        'rmsf':rmsf.cpu().numpy().astype(np.float32),
    }, {
        'qe_shape': qe.shape, 'qr_shape': qr.shape, 'qn_shape': qn.shape,
        'M_shape': M.shape,'dccm_shape': dccm.shape,'rmsf_shape': rmsf.shape,
    }

def pack_dataset_items(subunits, contacts, molecule_ids, max_num_nn, device=pt.device("cpu")):
    # prepare storage
    structures_data = {}
    contacts_data = {}

    # extract features and contacts for all subunits with contacts
    for cid0 in contacts:
        # get subunit
        s0 = subunits[cid0]
        s0 = concatenate_chains({cid0:s0})
        ID0 = s0['ID']

        # extract features, encode structure and compute topology
        qe0, qr0, qn0 = encode_features(s0,ID0)
        X0, M0 = encode_structure(s0, ID0, device=device)
        Xm0 = mean_coordinates(X0,ID0)
        ids0_topk = extract_topology(X0, ID0, max_num_nn)[0]
        dccm, rmsf = extract_dynamic_features(X0, ID0)

        # store structure data
        structures_data[cid0] = pack_structure_data(Xm0, qe0, qr0, qn0, M0, ids0_topk, dccm, rmsf)

        # prepare storage
        if cid0 not in contacts_data:
            contacts_data[cid0] = {}

        # for all contacting subunits
        for cid1 in contacts[cid0]:
            # prepare storage for swapped interface
            if cid1 not in contacts_data:
                contacts_data[cid1] = {}

            # if contacts not already computed
            if cid1 not in contacts_data[cid0]:
                # get contacting subunit
                s1 = subunits[cid1]
                s1 = concatenate_chains({cid1:s1})
                ID1 = s1['ID']

                # encode structure
                X1, M1 = encode_structure(s1, ID1, device=device)

                # nonzero not supported for array with more than I_MAX elements
                if (M0.shape[1] * M1.shape[1] * (molecule_ids.shape[0]**2)) > 2e9:
                    # compute interface targets
                    ctc_ids = contacts[cid0][cid1]['ids'].cpu()
                    freq = contacts[cid0][cid1]['freq'].cpu()
                    Y, T = contacts_types(s0, M0.cpu(), s1, M1.cpu(), ctc_ids, freq, molecule_ids, device=pt.device("cpu"))
                else:
                    # compute interface targets
                    ctc_ids = contacts[cid0][cid1]['ids'].to(device)
                    freq = contacts[cid0][cid1]['freq'].to(device)
                    Y, T = contacts_types(s0, M0.to(device), s1, M1.to(device), ctc_ids, freq, molecule_ids, device=device)

                # if has contacts of compatible type
                if pt.any(Y):
                    # store contacts data e.g., [ 1, 11, 13,  4] chain A res 1 (mtype 13) chain B res 11 (mtype 4)
                    contacts_data[cid0][cid1] = pack_contacts_data(Y, T)
                    contacts_data[cid1][cid0] = pack_contacts_data(Y.permute(1,0,3,2), T.transpose(0,1))

                # clear cuda cache
                pt.cuda.empty_cache()

    return structures_data, contacts_data

def store_dataset_items(hf, pdbid, structures_data, contacts_data):
    # metadata storage
    metadata_l = []

    # for all subunits with contacts
    for cid0 in contacts_data:
        # define store key
        key = f"{pdbid.upper()}/{cid0}"

        # save structure data
        hgrp = hf.create_group(f"data/structures/{key}")
        save_data(hgrp, attrs=structures_data[cid0][1], **structures_data[cid0][0])

        # for all contacting subunits
        for cid1 in contacts_data[cid0]:
            # define contacts store key
            ckey = f"{key}/{cid1}"

            # save contacts data
            hgrp = hf.create_group(f"data/contacts/{ckey}")
            save_data(hgrp, attrs=contacts_data[cid0][cid1][1], **contacts_data[cid0][cid1][0])

            # store metadata
            metadata_l.append({
                'key': key,
                'size': (np.max(structures_data[cid0][0]["M"], axis=0)+1).astype(int),
                'ckey': ckey,
                'ctype': contacts_data[cid0][cid1][1]["ctype"],
            })

    return metadata_l

def save_data(hgrp, attrs={}, **data):
    # store data
    for key in data:
        hgrp.create_dataset(key, data=data[key], compression="lzf")

    # save attributes
    for key in attrs:
        hgrp.attrs[key] = attrs[key]

if __name__ == "__main__":
    # set up dataset
    dataset = StructuresDataset(config_dataset['pdb_filepaths'], with_preprocessing=False)
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=16, pin_memory=False, prefetch_factor=4)
    
    # define device
    device = pt.device("cuda")

    # process structure, compute features and write dataset
    with h5py.File(config_dataset['dataset_filepath'], 'w', libver='latest') as hf:
        # store dataset encoding
        for key in config_encoding:
            hf[f"metadata/{key}"] = config_encoding[key].astype(np.string_)

        # save contact type encoding
        hf["metadata/mids"] = config_dataset['molecule_ids'].astype(np.string_)

        # prepare and store all structures
        metadata_l = []
        pbar = tqdm(dataloader)
        for structure, pdb_filepath in pbar:
            # check that structure was loaded
            if structure is None:
                continue

            # parse filepath
            m = re.match(r'.*/([a-z0-9]*)\.pdb', pdb_filepath)
            pdbid = m[1]
            print (pdbid)

            # check size
            if structure[0]['xyz'].shape[0] >= config_dataset['max_num_atoms']:
                continue

            # process structure
            structure = clean_structure(structure)

            # update molecules chains
            structure = tag_hetatm_chains(structure)

            # split structure
            subunits = split_by_chain(structure)

            # remove non atomic structures
            subunits = filter_non_atomic_subunits(subunits)

            # check not monomer
            if len(subunits) < 2:
                continue

            # remove duplicated molecules and ions
            subunits = remove_duplicate_tagged_subunits(subunits)
            
            # reorganize the format; conformer{chain{attribute}} -> chain{conformer{attribute}}
            subunits = reorganize_structure(subunits)

            # extract all contacts from assembly
            contacts = extract_all_contacts(subunits, config_dataset['r_thr'], device=device)

            # check there are contacts
            if len(contacts) == 0:
                continue
            
            # pack dataset items
            structures_data, contacts_data = pack_dataset_items(
                subunits, contacts,
                config_dataset['molecule_ids'],
                config_dataset['max_num_nn'], device=device
            )
            
            # store data
            metadata = store_dataset_items(hf, pdbid, structures_data, contacts_data)
            metadata_l.extend(metadata)
            
            # debug print
            pbar.set_description(f"{metadata_l[-1]['key']}: {metadata_l[-1]['size']}")
            
        # store metadata
        hf['metadata/keys'] = np.array([m['key'] for m in metadata_l]).astype(np.string_)
        hf['metadata/sizes'] = np.array([m['size'] for m in metadata_l])
        hf['metadata/ckeys'] = np.array([m['ckey'] for m in metadata_l]).astype(np.string_)
        hf['metadata/ctypes'] = np.stack(np.where(np.array([m['ctype'] for m in metadata_l])), axis=1).astype(np.uint32)
