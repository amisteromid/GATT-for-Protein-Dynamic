import torch as pt
import gemmi
import numpy as np
from gemmi import cif


def read_pdb(pdb_filepath):
    # read pdb and return information of all atoms in the pdb in dictionary format
    doc = gemmi.read_pdb(pdb_filepath, max_line_length=80)

    model_dic = {}
    
    # parse structure
    for mid, model in enumerate(doc):
		# altloc memory
        altloc_l = []
        icodes = []
        
		# data storage
        atom_element = []
        atom_name = []
        atom_xyz = []
        residue_name = []
        seq_id = []
        het_flag = []
        chain_name = []
        for a in model.all():
            # altloc check (keep first encountered)
            if a.atom.has_altloc():
                key = f"{a.chain.name}_{a.residue.seqid.num}_{a.atom.name}"
                if key in altloc_l:
                    continue
                else:
                    altloc_l.append(key)

            # insertion code (skip)
            icodes.append(a.residue.seqid.icode.strip())

            # store data
            atom_element.append(a.atom.element.name)
            atom_name.append(a.atom.name)
            atom_xyz.append([a.atom.pos.x, a.atom.pos.y, a.atom.pos.z])
            residue_name.append(a.residue.name)
            seq_id.append(a.residue.seqid.num)
            het_flag.append(a.residue.het_flag)
            chain_name.append(f"{a.chain.name}")
        model_dic[mid]={
        'xyz': np.array(atom_xyz, dtype=np.float32),
        'name': np.array(atom_name),
        'element': np.array(atom_element),
        'resname': np.array(residue_name),
        'resid': np.array(seq_id, dtype=np.int32),
        'het_flag': np.array(het_flag),
        'chain_name': np.array(chain_name),
        'icode': np.array(icodes)}

    # pack data
    return model_dic

def clean_structure(structure, rm_wat=True):
	# This function removes certain elements, then clean and reorganize the the structure
    for conformer in structure:
		# mask for water, hydrogens and deuterium
        m_wat = (structure[conformer]["resname"] == "HOH")
        m_h = (structure[conformer]["element"] == "H")
        m_d = (structure[conformer]["element"] == "D")
        m_hwat = (structure[conformer]["resname"] == "DOD")

        if rm_wat:
            # remove water
            mask = ((~m_wat) & (~m_h) & (~m_d) & (~m_hwat))
        else:
			# keep but tag water
            mask = ((~m_h) & (~m_d) & (~m_hwat))
            structure[conformer]["resid"][m_wat] = -999

        # filter structure atoms
        structure[conformer] = {key:structure[conformer][key][mask] for key in structure[conformer]}

        # find changes due to chain
        chains = structure[conformer]["chain_name"]
        ids_chains = np.where(np.array(chains).reshape(-1,1) == np.unique(chains).reshape(1,-1))[1]
        delta_chains = np.abs(np.sign(np.concatenate([[0], np.diff(ids_chains)])))

		# find changes due to inertion code
        icodes = structure[conformer]["icode"]
        ids_icodes = np.where(np.array(icodes).reshape(-1,1) == np.unique(icodes).reshape(1,-1))[1]
        delta_icodes = np.abs(np.sign(np.concatenate([[0], np.diff(ids_icodes)])))

        # find changes due to resids
        resids = structure[conformer]["resid"]
        delta_resids = np.abs(np.sign(np.concatenate([[0], np.diff(resids)])))

        # renumber resids
        resids = np.cumsum(np.sign(delta_chains + delta_resids + delta_icodes)) + 1

        # update resids
        structure[conformer]['resid'] = resids

		# remove uncessary icodes
        structure[conformer].pop("icode")

    # return process structure
    return structure

def tag_hetatm_chains(structure):
	# This function indetifies HET atoms, then update their chain names. e.g., A -> A:1
	
	for conformer in structure:
		# get hetatm
		m_hetatm = (structure[conformer]['het_flag'] == "H")
		resids_hetatm = structure[conformer]['resid'][m_hetatm]

		# split if multiple hetatm
		delta_hetatm = np.cumsum(np.abs(np.sign(np.concatenate([[0], np.diff(resids_hetatm)]))))

		# update chain name
		cids_hetatm = np.array([f"{cid}:{hid}" for cid, hid in zip(structure[conformer]['chain_name'][m_hetatm], delta_hetatm)])
		cids = structure[conformer]['chain_name'].copy().astype(np.dtype('<U10'))
		cids[m_hetatm] = cids_hetatm
		structure[conformer]['chain_name'] = np.array(list(cids)).astype(str)
	return structure


def atom_select(structure, sel):
    return {key: structure[key][sel] for key in structure}
    
    
def split_by_chain(structure):
	# Makes a new dictionary that the keys are the chain and then each chain would have same information as before
	
	new_structure = {}
	for conformer in structure:	
		# define storage
		chains = {}

		# define mask for chains
		cnames = structure[conformer]["chain_name"]
		ucnames = np.unique(cnames)
		m_chains = (cnames.reshape(-1,1) == np.unique(cnames).reshape(1,-1))

		# find all interfaces in biounit
		for i in range(len(ucnames)):
			# get chain
			chain = atom_select(structure[conformer], m_chains[:,i])
			chain.pop("chain_name")
			# store chain data
			chains[ucnames[i]] = chain
		new_structure[conformer] = chains
	
	return new_structure

def filter_non_atomic_subunits(subunits):
	# This removes the the subunits (~chains) that number of res = atoms
	
	for conformer in subunits:
		for sname in list(subunits[conformer]):
			n_res = np.unique(subunits[conformer][sname]['resid']).shape[0]
			n_atm = subunits[conformer][sname]['xyz'].shape[0]

			if (n_atm == n_res) & (n_atm > 1):
				subunits[conformer].pop(sname)

	return subunits

def remove_duplicate_tagged_subunits(subunits):
	# Remove duplicated HETATOM subunits
	
	for conformer in subunits: 	
		# located tagged HETATOM subunits
		tagged_cids = [cid for cid in subunits[conformer] if (len(cid.split(':')) == 3)]
		# remove if overlapping
		for i in range(len(tagged_cids)):
			cid_i = tagged_cids[i]
			for j in range(i+1, len(tagged_cids)):
				cid_j = tagged_cids[j]

				# check if still existing
				if (cid_i in subunits[conformer]) and (cid_j in subunits[conformer]):
					# extract distances
					xyz0 = subunits[conformer][cid_i]['xyz']
					xyz1 = subunits[conformer][cid_j]['xyz']

					# if same size
					if xyz0.shape[0] == xyz1.shape[0]:
						# minimum self distances
						d_min = np.min(np.linalg.norm(xyz0 - xyz1, axis=1))
						if d_min < 0.2:
							subunits[conformer].pop(cid_j)

	return subunits

def reorganize_structure(structure):
	# Converts a dic of conformer, chain, attributes to chain, conformer, attributes
    new_structure = {}

    for conformer, chains in structure.items():
        for chain, atom_attributes in chains.items():
            if chain not in new_structure:
                new_structure[chain] = {}

            new_structure[chain][conformer] = atom_attributes
    return new_structure

class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, pdb_filepaths, with_preprocessing=True):
        super(StructuresDataset).__init__()
        # store dataset filepath
        self.pdb_filepaths = pdb_filepaths

        # store flag
        self.with_preprocessing = with_preprocessing

    def __len__(self):
        return len(self.pdb_filepaths)

    def __getitem__(self, i):
        # find pdb filepath
        pdb_filepath = self.pdb_filepaths[i]

        # parse pdb
        try:
            structure = read_pdb(pdb_filepath)
        except Exception as e:
            print(f"ReadError: {pdb_filepath}: {e}")
            return None, pdb_filepath

        if self.with_preprocessing:
            # process structure (water and hydrogen)
            structure = clean_structure(structure, rm_wat=True)

            # update molecules chains
            structure = tag_hetatm_chains(structure)

            # split structure
            subunits = split_by_chain(structure)

            # remove non atomic structures
            subunits = filter_non_atomic_subunits(subunits)

            # remove duplicated molecules and ions
            subunits = remove_duplicate_tagged_subunits(subunits)
            
            # reorganize the format; conformer{chain{attribute}} -> chain{conformer{attribute}}
            subunits = reorganize_structure(subunits)
            
            return subunits, pdb_filepath
        else:
            return structure, pdb_filepath


#structure_6wa1 = StructuresDataset(['data/6wa1.pdb'],  with_preprocessing=True)[0][0]
#print (structure_6wa1['A'].keys())

def remove_hetatm_chains(structure):
	
	for chain in list(structure.keys()):
		if ':' in chain:
			del structure[chain]
	return structure

def concatenate_chains(chains):
	# get intersection of keys between chains
	keys = set.intersection(*[set(conformer.keys()) for chain in chains.values() for conformer in chain.values()])
	
	# concatenate chains
	structure = {key: np.concatenate([chains[chain][conformer][key] for chain in chains.keys() for conformer in chains[chain].keys()]) for key in keys}

	# add atom information
	structure['ID'] = np.concatenate([np.array([str(conformerid)+'_'+cid]*chains[cid][conformerid]['xyz'].shape[0]) for cid in chains.keys() for conformerid in chains[cid].keys()])
	return structure


