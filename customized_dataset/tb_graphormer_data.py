import pandas as pd
from graphormer.data import register_dataset
from sklearn.model_selection import train_test_split
from argparse import Namespace



import os
import time
from typing import List


import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, Data)
import pandas as pd
import numpy as np


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT


class MolecularGraph:
    molecule: rdkit.Chem.rdchem.Mol

    types: dict
    bonds: dict
    atom_features: dict
    bond_features: dict

    pos: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    ascii_name: torch.Tensor
    external_input: torch.Tensor

    def __init__(self, molecule: rdkit.Chem.rdchem.Mol, pos: torch.Tensor, y: torch.Tensor, num_atoms: int,
                 external_input: torch.Tensor):
        """
        Helper class to easily build a molecular graph from a rdkit.Chem.rdchem.Mol type molecule
        Args:
            molecule: rdkit.Chem.rdchem.Mol type molecule for feature extraction
            pos: torch.Tensor containing 3D positional information
            y: torch.Tensor of targets
            num_atoms: int number of atoms
            external_input: additional input data
        """
        self.atom_features = {
            # atom features
            "type_idx": [],
            "aromatic": [],
            "ring": [],
            "sp": [],
            "sp2": [],
            "sp3": [],
            "sp3d": [],
            "sp3d2": [],
            "num_hs": [],
            "num_neighbors": []
        }
        self.bond_features = {
            # bond features
            "row": [],
            "col": [],
            "bond_idx": [],
            "conj": [],
            "ring": [],
            "stereo": []
        }
        if rdkit is not None:
            # atom types
            self.types = {
                'C': 0,
                'O': 1,
                'H': 2,
                'N': 3,
                'F': 4,
                'Si': 5,
                'P': 6,
                'S': 7,
                'Cl': 8,
                'Ti': 9,
                'Cu': 10,
                'Br': 11,
                'Sn': 12,
                'I': 13,
                'Bi': 14

            }
            # bond types
            self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

            self.external_input = external_input

            self.__prepare_graph(molecule, pos, y, num_atoms)

    def set_molecule(self, molecule: rdkit.Chem.rdchem.Mol) -> None:
        self.molecule = molecule

    def __set_atom_features(self) -> None:
        for atom in self.molecule.GetAtoms():
            self.atom_features["type_idx"].append(self.types[atom.GetSymbol()])
            self.atom_features["aromatic"].append(1 if atom.GetIsAromatic() else 0)
            self.atom_features["ring"].append(1 if atom.IsInRing() else 0)
            hybridization = atom.GetHybridization()
            self.atom_features["hybridization"] = hybridization
            self.atom_features["sp"].append(1 if hybridization == HybridizationType.SP else 0)
            self.atom_features["sp2"].append(1 if hybridization == HybridizationType.SP2 else 0)
            self.atom_features["sp3"].append(1 if hybridization == HybridizationType.SP3 else 0)
            if (hybridization == HybridizationType.SP3D) or (hybridization == HybridizationType.SP3D):
                raise ValueError(f'{hybridization} type not incldued in preprocessing. Please add.')
            # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)
            # sp3d2.append(1 if hybridization == HybridizationType.SP3D2 else 0)
            self.atom_features["num_hs"].append(atom.GetTotalNumHs(includeNeighbors=True))
            self.atom_features["num_neighbors"].append(len(atom.GetNeighbors()))

    def __set_bond_features(self) -> None:
        for bond in self.molecule.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            self.bond_features["row"] += [start, end]
            self.bond_features["col"] += [end, start]
            self.bond_features["bond_idx"] += 2 * [self.bonds[bond.GetBondType()]]
            self.bond_features["conj"].append(bond.GetIsConjugated())
            self.bond_features["conj"].append(bond.GetIsConjugated())
            self.bond_features["ring"].append(bond.IsInRing())
            self.bond_features["ring"].append(bond.IsInRing())
            self.bond_features["stereo"].append(bond.GetStereo())
            self.bond_features["stereo"].append(bond.GetStereo())

    def __set_edges(self, num_atoms: int) -> None:
        edge_index = torch.tensor([self.bond_features["row"], self.bond_features["col"]], dtype=torch.long)
        e1 = F.one_hot(torch.tensor(self.bond_features["bond_idx"]), num_classes=len(self.bonds)).to(torch.float)
        e2 = torch.tensor([self.bond_features["conj"], self.bond_features["ring"]], dtype=torch.float).t().contiguous()
        e3 = F.one_hot(torch.tensor(self.bond_features["stereo"]), num_classes=4).to(torch.float)
        edge_attr = torch.cat([e1, e2, e3], dim=-1).to(torch.long)
        self.edge_index, self.edge_attr = coalesce(edge_index, edge_attr, num_atoms, num_atoms)

    def __set_x(self) -> None:
        x1 = F.one_hot(torch.tensor(self.atom_features["type_idx"]), num_classes=len(self.types))
        x2 = torch.tensor([self.atom_features["aromatic"], self.atom_features["ring"], self.atom_features["sp"],
                           self.atom_features["sp2"], self.atom_features["sp3"]], dtype=torch.float).t().contiguous()
        x4 = F.one_hot(torch.tensor(self.atom_features["num_hs"]), num_classes=5)
        x = torch.cat([x1.to(torch.float), x2, x4.to(torch.float)], dim=-1)
        self.x = x

    def __set_pos(self, pos: torch.Tensor) -> None:
        self.pos = pos

    def __set_y(self, y: torch.Tensor) -> None:
        self.y = y

    def __set_ascii_name(self) -> None:
        # Set name manually or calculate using RDKIT?
        # name = str(Chem.MolToSmiles(self.molecule))
        name = self.molecule.GetProp("SMILES")
        self.ascii_name = name

    def __prepare_graph(self, molecule: rdkit.Chem.rdchem.Mol, pos: torch.Tensor, y: torch.Tensor, num_atoms: int) \
            -> None:
        self.set_molecule(molecule)
        self.__set_atom_features()
        self.__set_bond_features()
        self.__set_edges(num_atoms)
        self.__set_pos(pos)
        self.__set_x()
        self.__set_y(y)
        self.__set_ascii_name()

    def graph_to_data(self) -> Data:
        data = Data(x=self.x, pos=self.pos, edge_index=self.edge_index,
                    edge_attr=self.edge_attr, y=self.y, mol_id=self.ascii_name)
        data.external_input = self.external_input
        return data


class SmilesToGraph(InMemoryDataset):
    raw_dataframe: pd.DataFrame

    raw_url = ''
    processed_url = ''

    def __init__(self, root, parameters, transform=None, pre_transform=None,
                 pre_filter=None):
        """
        Dataclass that automatically converts a dataset containing SMILES to molecular graphs. The data is processed
        once and can then be retrieved without processing again.
        Args:
            root: root directory
            parameters: dataset parameters
            transform:
            pre_transform:
            pre_filter:
        """
        self.parameters = parameters
        super().__init__(root, transform, pre_transform, pre_filter)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.targets = parameters.target_properties

    @property
    def raw_file_names(self):
        # TODO implement for sql
        return ['raw.csv', 'raw.json', 'raw.xlsx']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def __set_dataframe(self) -> None:
        data_path_list = self.raw_paths
        usable_data_path = ""
        # check which of the possible data types is provided
        for data_path in data_path_list:
            if os.path.isfile(data_path):
                usable_data_path = data_path

        if usable_data_path == "":
            raise ValueError("No data provided!")

        else:
            if usable_data_path.endswith("csv"):
                self.raw_dataframe = pd.read_csv(usable_data_path, sep=None)
            elif usable_data_path.endswith("json"):
                self.raw_dataframe = pd.read_json(usable_data_path)
            elif usable_data_path.endswith("xlsx"):
                self.raw_dataframe = pd.read_excel(usable_data_path)
            else:
                raise NotImplementedError("File type is not supported. Please provide csv, json, sql or xlsx files.")

            self.raw_dataframe.fillna("inf")

    def __get_header_names(self) -> List[str]:
        """
        private function to return the headers as a list of strings
        :return: list of header names
        """

        header_names = self.raw_dataframe.columns.values.tolist()
        return header_names

    def __get_molecules(self) -> List[str]:
        """
        private function to collect every molecule based on the "SMILES" index and the data table
        :return: list of SMILE strings
        """

        molecules = self.raw_dataframe["SMILES"].values.tolist()
        return molecules

    def __write_sdf_file(self) -> None:
        """
        private function to write every molecule into a sdf file
        :return: None
        """
        print("Writing sdf files...")
        writer = Chem.SDWriter(str(self.root) + '/raw/raw.sdf')
        molecules = self.__get_molecules()
        for m in molecules:
            print(m)
            mol = Chem.rdmolfiles.MolFromSmiles(m)
            # Set molecule property that contains the SMILES string.
            mol.SetProp("SMILES", m)
            writer.write(mol)
        del writer

        # delay for proper saving of sdf file
        time.sleep(10)

    def __get_target_data(self) -> torch.Tensor:
        """
        private function to collect the target data
        :return: tensor containing the target data
        """

        target_data = [row for row in self.raw_dataframe[self.parameters.target_properties].values]
        # added for faster conversion
        target_data = np.array(target_data)
        target_data = torch.tensor(target_data, dtype=torch.float)
        return target_data

    def __get_external_input_data(self, mol_idx: int):
        """
        private function to extract external input data if given. Else, None is returned
        :return: external_input_data as torch.Tensor or None
        """
        external_input_data = None
        if len(self.parameters.additional_MLP_input_cols) > 0:
            external_input_data = self.raw_dataframe[self.parameters.additional_MLP_input_cols].values[mol_idx].reshape(1, -1)
            external_input_data = torch.tensor(external_input_data, dtype=torch.float)

        return external_input_data

    def __get_molecular_graphs(self) -> List[Data]:
        """
        private function to transform SMILES into graph data
        :return: list of graph data with graph data in form of a torch_geometric.data.Data object
        """
        dataset = str(self.root) + '/raw/raw.sdf'
        suppl = Chem.SDMolSupplier(dataset, removeHs=False)
        targets = self.__get_target_data()

        if len(targets) != len(suppl):
            raise ValueError('Number of target data points does not match number of molecules')

        data_list = []
        for i, mol in enumerate(suppl):
            if mol is None:
                print('Invalid molecule (None)')
                continue

            text = suppl.GetItemText(i)

            num_atoms = mol.GetNumAtoms()

            # only consider molecules with more than one atom -> need at least one bond for a valid graph
            if num_atoms < 1:
                print('Warning: molecule skipped because it contains no atoms')
                continue

            # spatial positions of atom (Note: not included in training, can be used for future implementations)
            pos = text.split('\n')[4:4 + num_atoms]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            y = targets[i].unsqueeze(0)

            ext_input = self.__get_external_input_data(mol_idx=i)

            temp_molecular_graph = MolecularGraph(molecule=mol, pos=pos, y=y, num_atoms=num_atoms,
                                                  external_input=ext_input)
            data = temp_molecular_graph.graph_to_data()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        return data_list

    def __apply_pre_filter(self, data_list: Data) -> Data:

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        return data_list

    def __apply_pre_transform(self, data_list: Data) -> Data:

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return data_list

    def __apply_preprocessing(self):
        self.data, self.slices = torch.load(self.processed_paths[0])
        data_list = [data for data in self]

        data_list = self.__apply_pre_filter(data_list)
        data_list = self.__apply_pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def process(self) -> None:
        """
        private function to process given data into graph data
        :return:
        """
        # if processed data has not yet been created:
        if not os.path.isfile(self.processed_paths[0]):
            print(f"Processing data in {self.root}")
            self.__set_dataframe()
            self.__write_sdf_file()
            print("Finished!")

            print("Fetching data...")
            data_list = self.__get_molecular_graphs()
            torch.save(self.collate(data_list), self.processed_paths[0])
            print(f"Saved processed data in {self.processed_paths[0]}")

        else:
            print("Using processed version of the dataset!")

        self.__apply_preprocessing()


class MyFilter:
    def __call__(self, data):
        return data.num_nodes > 1  # Remove graphs with less than 2 nodes.


class MyPreTransform:
    def __call__(self, data):
        return data



@register_dataset("Tb_dataset")
def create_customized_dataset():

    parameters = {

        "target_data_dir": "Tb",
        "target_properties": ['Tb'],
        "save_dir_id": "",
        "additional_MLP_input_cols": []
    }

    parameters = Namespace(**parameters)

    print(os.getcwd())

    dataset = SmilesToGraph("./customized_dataset/Tb/Total", parameters)
    train_data = pd.read_csv("./customized_dataset/Tb/Train/raw/raw.csv", sep=";")
    test_data = pd.read_csv("./customized_dataset/Tb/Test/raw/raw.csv", sep=";")

    train_idx, valid_idx = train_test_split(train_data["index"].tolist(), shuffle=True, test_size=0.15)
    train_idx = np.array(train_idx).astype(int)
    valid_idx = np.array(valid_idx).astype(int)

    test_idx = np.array(test_data["index"].tolist()).astype(int)

    print(
        {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg"
    }
    )

    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg"
    }
