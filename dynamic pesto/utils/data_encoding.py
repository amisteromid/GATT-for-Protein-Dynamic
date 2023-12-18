import numpy as np
import torch as pt


# standard elements (sorted by aboundance) (32)
std_elements = np.array([
    'C', 'O', 'N', 'S', 'P', 'Se', 'Mg', 'Cl', 'Zn', 'Fe', 'Ca', 'Na',
    'F', 'Mn', 'I', 'K', 'Br', 'Cu', 'Cd', 'Ni', 'Co', 'Sr', 'Hg', 'W',
    'As', 'B', 'Mo', 'Ba', 'Pt'
])

# standard residue names: AA/RNA/DNA (sorted by aboundance) (29)
std_resnames = np.array([
    'LEU', 'GLU', 'ARG', 'LYS', 'VAL', 'ILE', 'PHE', 'ASP', 'TYR',
    'ALA', 'THR', 'SER', 'GLN', 'ASN', 'PRO', 'GLY', 'HIS', 'TRP',
    'MET', 'CYS', 'G', 'A', 'C', 'U', 'DG', 'DA', 'DT', 'DC'
])

# standard atom names contained in standard residues (sorted by aboundance) (63)
std_names = np.array([
    'CA', 'N', 'C', 'O', 'CB', 'CG', 'CD2', 'CD1', 'CG1', 'CG2', 'CD',
    'OE1', 'OE2', 'OG', 'OG1', 'OD1', 'OD2', 'CE', 'NZ', 'NE', 'CZ',
    'NH2', 'NH1', 'ND2', 'CE2', 'CE1', 'NE2', 'OH', 'ND1', 'SD', 'SG',
    'NE1', 'CE3', 'CZ3', 'CZ2', 'CH2', 'P', "C3'", "C4'", "O3'", "C5'",
    "O5'", "O4'", "C1'", "C2'", "O2'", 'OP1', 'OP2', 'N9', 'N2', 'O6',
    'N7', 'C8', 'N1', 'N3', 'C2', 'C4', 'C6', 'C5', 'N6', 'N4', 'O2',
    'O4'
])

# resname categories
categ_to_resnames = {
    "protein": ['GLU', 'LEU', 'ALA', 'ASP', 'SER', 'VAL', 'GLY', 'THR', 'ARG',
                'PHE', 'TYR', 'ILE', 'PRO', 'ASN', 'LYS', 'GLN', 'HIS', 'TRP',
                'MET', 'CYS'],
    "rna": ['A', 'U', 'G', 'C'],
    "dna": ['DA', 'DT', 'DG', 'DC'],
    "ion": ['MG', 'ZN', 'CL', 'CA', 'NA', 'MN', 'K', 'IOD', 'CD', 'CU', 'FE', 'NI',
            'SR', 'BR', 'CO', 'HG'],
    "ligand": ['SO4', 'NAG', 'PO4', 'EDO', 'ACT', 'MAN', 'HEM', 'FMT', 'BMA',
               'ADP', 'FAD', 'NAD', 'NO3', 'GLC', 'ATP', 'NAP', 'BGC', 'GDP',
               'FUC', 'FES', 'FMN', 'GAL', 'GTP', 'PLP', 'MLI', 'ANP', 'H4B',
               'AMP', 'NDP', 'SAH', 'OXY'],
    "lipid": ['PLM', 'CLR', 'CDL', 'RET'],
}
resname_to_categ = {rn:c for c in categ_to_resnames for rn in categ_to_resnames[c]}

# prepare back mapping
elements_enum = np.concatenate([std_elements, [b'X']])
names_enum = np.concatenate([std_names, [b'UNK']])
resnames_enum = np.concatenate([std_resnames, [b'UNX']])

# prepare config summary
config_encoding = {'std_elements': std_elements, 'std_resnames': std_resnames, 'std_names': std_names}


def onehot(x, v):
    m = (x.reshape(-1,1) == np.array(v).reshape(1,-1))
    return np.concatenate([m, ~np.any(m, axis=1).reshape(-1,1)], axis=1)


def encode_structure(structure, chain_name, device=pt.device("cpu")):
    n_chain_name = len(np.unique(chain_name))
    try:
        n_atoms = int(len(chain_name)/n_chain_name)
    except ValueError:
        # If the conversion fails, the value is not an integer
        print("The conformer number doesn't match the number of atoms.")
        return None
    
    # coordinates
    if isinstance(structure['xyz'], pt.Tensor):
        X = structure['xyz'].to(device)
    else:
        X = pt.from_numpy(structure['xyz'].astype(np.float32)).to(device)

    # residues mapping for reduce
    if isinstance(structure['resid'], pt.Tensor):
        resids = structure['resid'].to(device)
    else:
        resids = pt.from_numpy(structure['resid']).to(device)
    M = (resids.unsqueeze(1) == pt.unique(resids).unsqueeze(0))
    
    # merge residue types to remove conformer dimension
    number_of_chain = len(np.unique(list(map(lambda x: x.split('_')[1], chain_name))))
    if number_of_chain == 1:
        M = M.view(n_chain_name, n_atoms, len(pt.unique(resids)))
        for j in range(M.shape[0]):
            if not pt.equal(M[j, :, :], M[0, :, :]):
                print ("somthing wrong in resid")
                return
	
        return X, M[0,:,:]
    else:
        print (f'{number_of_chain}-chain structure')
        return

def mean_coordinates(X, chain_name):
	n_chain_name = len(np.unique(chain_name))
	n_atoms = int(len(chain_name)/n_chain_name)
	# Reorganize the X
	reorganized_X = X.view(n_chain_name, n_atoms, 3)
	# mean of coordiantes
	Xm = pt.mean(reorganized_X, dim=0)
	return Xm

def encode_features(structure, chain_name, device=pt.device("cpu")):
    n_chain_name = len(np.unique(chain_name))
    try:
        n_atoms = int(len(chain_name)/n_chain_name)
    except ValueError:
        # If the conversion fails, the value is not an integer
        print("The conformer number doesn't match the number of atoms.")
        return None
        
    # charge features
    qe = pt.from_numpy(onehot(structure['element'], std_elements).astype(np.float32)).to(device)
    qr = pt.from_numpy(onehot(structure['resname'], std_resnames).astype(np.float32)).to(device)
    qn = pt.from_numpy(onehot(structure['name'], std_names).astype(np.float32)).to(device)
    # reorganize and merge
    qe = qe.view(n_chain_name, n_atoms, qe.shape[1])
    qr = qr.view(n_chain_name, n_atoms, qr.shape[1])
    qn = qn.view(n_chain_name, n_atoms, qn.shape[1])
    for j in range(qe.shape[0]):
        if not pt.equal(qe[j, :, :], qe[0, :, :]):
            print ("somthing wrong in element names")
            return
    return qe[0,:,:],qr[0,:,:],qn[0,:,:]

def extract_topology(X, chain_name, num_nn):
    n_chain_name = len(np.unique(chain_name))
    n_atoms = int(len(chain_name)/n_chain_name)
    
    # Reorganize the X
    reorganized_X = X.view(n_chain_name, n_atoms, 3)
    
    # Initialize a tensor to store all distance matrices
    all_distance_matrices = pt.zeros((n_chain_name, n_atoms, n_atoms))

    # Compute the distance matrix for each chain
    for i in range(n_chain_name):
        # Select the ith chain
        chain_X = reorganized_X[i]
        # Compute the pairwise distance matrix for the ith chain
        R = chain_X.unsqueeze(0) - chain_X.unsqueeze(1)
        D = pt.norm(R, dim=2)
        # Store the distance matrix in the all_distance_matrices tensor
        all_distance_matrices[i] = D

    # Compute the average distance matrix
    average_distance_matrix = pt.mean(all_distance_matrices, dim=0)
    # Not to consider very small distances 
    D = D + pt.max(D)*(D < 1e-2).float()
    # normalize displacement vectors
    R = R / D.unsqueeze(2)

    # find nearest neighbors
    knn = min(num_nn, D.shape[0])
    D_topk, ids_topk = pt.topk(D, knn, dim=1, largest=False)
    R_topk = pt.gather(R, 1, ids_topk.unsqueeze(2).repeat((1,1,X.shape[1])))

    return ids_topk, D_topk, R_topk, D, R

def extract_dynamic_features(X, chain_name):
	
    n_chain_name = len(np.unique(chain_name))
    try:
        n_atoms = int(len(chain_name)/n_chain_name)
    except ValueError:
        # If the conversion fails, the value is not an integer
        print("The conformer number doesn't match the number of atoms.")
        return None
      
    # Reorganize the X
    reorganized_X = X.view(n_chain_name, n_atoms, 3)
    reorganized_chain_name = chain_name.reshape((n_chain_name, n_atoms))
    for index, name in enumerate(reorganized_chain_name):
        if len(np.unique(name)) > 1:
            print(f"Error; not all atoms belong to the same chain in index {index}")
            print(np.unique(name))
            return None
    # Calculate the mean position
    mean_pos = pt.mean(reorganized_X, dim=0)
    # Calculate displacements
    displacements = reorganized_X - mean_pos
    # Normalize the displacement vectors
    norms = pt.sqrt(pt.sum(displacements**2, axis=2, keepdim=True))
    normalized_displacements = displacements / norms
        
    # Initialize
    dccm = np.zeros((n_atoms, n_atoms))
    rmsf = np.zeros(n_atoms)
    # Calculate the dccm and rmsf
    for i in range(n_atoms):
        for j in range(n_atoms):
            corr = pt.sum(normalized_displacements[:, i, :] * normalized_displacements[:, j, :], axis=1)
            dccm[i, j] = pt.mean(corr).item()
            
        # RMSF calculation
        rmsf[i] = pt.sqrt(pt.mean(pt.sum(displacements[:, i, :]**2, axis=1))).item()

            
    return pt.from_numpy(dccm), pt.from_numpy(rmsf)
    
    
def pool_to_res_level(dccm, M):
    num_residues = M.shape[1]
    residue_dccm = pt.zeros((num_residues, num_residues), dtype=dccm.dtype, device=dccm.device)

    for i in range(num_residues):
        for j in range(num_residues):
            # Find the atoms that belong to each residue pair
            atoms_i = M[:, i].bool()
            atoms_j = M[:, j].bool()

            # Extract the corresponding submatrix from the DCCM
            submatrix = dccm[atoms_i, :][:, atoms_j]

            # Compute the average of the submatrix and assign it to the residue DCCM
            residue_dccm[i, j] = pt.mean(submatrix)
    return residue_dccm
