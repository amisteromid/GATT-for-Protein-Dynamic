import numpy as np
from glob import glob

config_dataset = {
    # parameters
    "r_thr": 5.0,  # Angstroms
    "max_num_atoms": 1024*8,
    "max_num_nn": 64,
    "molecule_ids": np.array([
        'GLU', 'LEU', 'ALA', 'ASP', 'SER', 'VAL', 'GLY', 'THR', 'ARG', 'PHE', 'TYR', 'ILE',
        'PRO', 'ASN', 'LYS', 'GLN', 'HIS', 'TRP', 'MET', 'CYS', 'A', 'U', 'G', 'C', 'DA',
        'DT', 'DG', 'DC', 'MG', 'ZN', 'CL', 'CA', 'NA', 'MN', 'K', 'IOD', 'CD', 'CU', 'FE',
        'NI', 'SR', 'BR', 'CO', 'HG', 'SO4', 'NAG', 'PO4', 'EDO', 'ACT', 'MAN', 'HEM', 'FMT',
        'BMA', 'ADP', 'FAD', 'NAD', 'NO3', 'GLC', 'ATP', 'NAP', 'BGC', 'GDP', 'FUC', 'FES',
        'FMN', 'GAL', 'GTP', 'PLP', 'MLI', 'ANP', 'H4B', 'AMP', 'NDP', 'SAH', 'OXY', 'PLM',
        'CLR', 'CDL', 'RET'
    ]),

    # input filepaths
    "pdb_filepaths": glob("/home/omokhtar/Desktop/Codes/data/*.pdb"),
    # "pdb_filepaths": glob(f"/tmp/{sys.argv[-1]}/all_biounits/*/*.pdb[0-9]*.gz"),

    # output filepath
    "dataset_filepath": "/home/omokhtar/Desktop/Codes/data/tmp.h5",
    # "dataset_filepath": f"/tmp/{sys.argv[-1]}/contacts_rr5A_64nn_8192.h5",
}


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
# prepare config summary
config_encoding = {'std_elements': std_elements, 'std_resnames': std_resnames, 'std_names': std_names}
