# Posted in X.
# https://x.com/Flasushi/status/1791622447315460540

from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import numpy as np

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    num_h_donors = Descriptors.NumHDonors(mol)
    num_h_acceptors = Descriptors.NumHAcceptors(mol)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_arr = np.zeros((1,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)

    descriptors = np.concatenate(([mol_weight, logp, num_h_donors, num_h_acceptors], fp_arr))

    return descriptors


if __name__ == '__main__':
    pass
