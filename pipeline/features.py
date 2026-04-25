import csv
import pickle

import networkx as nx
import numpy as np
from rdkit import Chem
from tqdm.auto import tqdm


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown"],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    )


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    graph = nx.Graph(edges).to_directed()
    edge_index = []
    for edge_start, edge_end in graph.edges:
        edge_index.append([edge_start, edge_end])

    return c_size, features, edge_index


def load_drug_smile(path="data/smiles.csv"):
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)

        drug_dict = {}
        drug_smile = []
        for name, smile, *_ in reader:
            if name not in drug_dict:
                drug_dict[name] = len(drug_dict)
            drug_smile.append(smile)

    smile_graph = {}
    for smile in drug_smile:
        smile_graph[smile] = smile_to_graph(smile)

    return drug_dict, drug_smile, smile_graph


def save_cell_oge_matrix(ge_frame):
    cell_dict = {}
    use_index = ge_frame.index.name in ["GENE_SYMBOLS", "Cosmic sample Id"] or "Cosmic sample Id" not in ge_frame.columns

    if use_index:
        iterator = ge_frame.iterrows()
        for cell_name, row in tqdm(iterator, total=len(ge_frame)):
            cell_dict[int(cell_name)] = np.asarray(row, dtype=float)
        return cell_dict

    iterator = ge_frame.iterrows()
    for _, row in tqdm(iterator, total=len(ge_frame)):
        cell_name = int(row["Cosmic sample Id"])
        cell_feature = np.asarray(row.drop(labels=["Cosmic sample Id"]), dtype=float)
        cell_dict[cell_name] = cell_feature
    return cell_dict


def save_cell_meth_matrix(path="data/gdsc/METH_CELLLINES_BEMs_PANCAN.csv"):
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)

        cell_dict = {}
        for item in reader:
            cell_id = int(item[0])
            cell_dict[cell_id] = np.asarray([int(value) for value in item[1:]])

    return cell_dict


def save_cell_mut_matrix(path="data/gdsc/PANCANCER_Genetic_feature.csv", mut_dict_path="mut_dict"):
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)

        cell_dict = {}
        mut_dict = {}
        matrix_list = []

        for item in reader:
            cell_id = int(item[1])
            mutation_name = item[5]
            is_mutated = int(item[6])

            if mutation_name not in mut_dict:
                mut_dict[mutation_name] = len(mut_dict)
            if cell_id not in cell_dict:
                cell_dict[cell_id] = len(cell_dict)

            if is_mutated == 1:
                matrix_list.append((cell_dict[cell_id], mut_dict[mutation_name]))

    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))
    for row_index, column_index in matrix_list:
        cell_feature[row_index, column_index] = 1

    with open(mut_dict_path, "wb") as handle:
        pickle.dump(mut_dict, handle)

    return cell_dict, cell_feature


def load_feature_state(ge_frame):
    cell_dict_mut, cell_feature_mut = save_cell_mut_matrix()
    cell_dict_meth = save_cell_meth_matrix()
    cell_dict_ge = save_cell_oge_matrix(ge_frame)
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    return {
        "cell_dict_mut": cell_dict_mut,
        "cell_feature_mut": cell_feature_mut,
        "cell_dict_meth": cell_dict_meth,
        "cell_dict_ge": cell_dict_ge,
        "drug_dict": drug_dict,
        "drug_smile": drug_smile,
        "smile_graph": smile_graph,
    }
