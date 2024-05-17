import os
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from functools import partial
from random import randint, shuffle
from typing import Optional, Callable, Literal, List, Any, Dict

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from biotite.structure.io import pdb


def exists(x: Any) -> bool:
    """Returns True if and only if x is not None"""
    return x is not None


def default(x: Any, y: Any) -> Any:
    """Returns x if x exists, otherwise y"""
    x = x if exists(x) else y


def identity(x):
    return x


class PinderDataset(torch.utils.data.Dataset):
    """Pinder Dataset, adapted copy from PDBDataset from Moleculearn. Only load the holo monomers pdbs"""

    def __init__(
        self,
        decoy_pdb_folder: str,
        target_pdb_folder: str,
        raise_exceptions: bool,
        split: str,
        use_bound: bool = False,
        dataframe_path: Optional[str] = None,
        align_seqs_to: str = "decoy",
        filter_fn: Optional[Any] = None,
        crop_len: int = -1,
        max_seq_len: int = -1,
        kabsch_align_chains: bool = True,
        load_monomer_ty: Literal["apo", "holo", "predicted", "all"] = "holo",
        limit_clusters: int = -1,
    ):
        """PDB Dataset

        Args:
            decoy_pdb_folder (str): root directory containing all decoy
                pdb files.

            target_pdb_folder (str): root directory containing all target
                pdb files.

            pdb_list_path (Optional[str]): path to list of pdb files to load
                (see `dataset_utils.load_pdb_list` for a description)

            cluster_list_path (Optional[str]): path to list of cluster files to load
                (see `dataset_utils.load_cluster_list` for a description)

            cluster_folder (Optional[str]): root directory containing cluster files

            raise_exceptions (bool): whether to raise or ignore exceptions

            align_seqs_to (str, optional): whether to align examples to sequences
                derived from "target" or "decoy" pdbs. Defaults to "target".

            filter_fn (Optional[Any], optional): function taking an example as
                input, and returning a boolean indicating whether the example is valid
                and should be passed ot the model. Defaults to None.
        """
        super().__init__()
        assert exists(dataframe_path)
        self.model_list = ModelListDf(
            target_pdb_folder=target_pdb_folder,
            decoy_pdb_folder=decoy_pdb_folder,
            df_path=dataframe_path,
            split=split,
            use_bound=use_bound,
            monomer_ty=load_monomer_ty,
            limit_clusters=limit_clusters,
        )
        self.raise_exceptions = raise_exceptions
        self.target_folder = target_pdb_folder
        self.decoy_folder = decoy_pdb_folder
        self.filter_fn = default(filter_fn, identity)
        self.align_seq_to = align_seqs_to
        self.crop_len = crop_len
        self.kabsch_align_chains = kabsch_align_chains
        self.max_seq_len = max_seq_len
        self.load_monomer_ty = load_monomer_ty

    def subset(self, start, end):
        if isinstance(start, float) and end <= 1:
            start = int(start * len(self))
            end = int(end * len(self))
        self.model_list = self.model_list.restrict(start, end)
        return self

    def __len__(self):
        return len(self.model_list)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        example = None
        while example is None:
            (
                decoy_pdb_paths,
                target_pdb_paths,
                decoy_chain_ids,
                target_chain_ids,
                exceptions,
                metadata,
            ) = self.model_list[idx]

            # optionally raise exceptions
            if len(exceptions) > 0:
                msgs = [str(e) for e in exceptions]
                print(f"[ERROR] caught exceptions {msgs} loading data")
                if self.raise_exceptions:
                    raise exceptions[0]
            try:
                example = self.load_example(
                    decoy_pdb_paths,
                    target_pdb_paths,
                    decoy_chain_ids,
                    target_chain_ids,
                    metadata,
                )
                example = example if self.filter_fn(example) else None

            except Exception as e:  # noqa
                print(f"[Warning] Got exception : {e} in dataloader")
                if self.raise_exceptions:
                    raise e
            # load another example at random
            idx = randint(0, len(self) - 1)

        example = align_chains(example) if self.kabsch_align_chains else example
        return example

    def load_example(
        self,
        decoy_pdb_paths: List[Path],
        target_pdb_paths: List[Path],
        decoy_chain_ids: List[str],
        target_chain_ids: List[str],
        metadata: Dict,
    ) -> Dict:
        """
        Loads a single example from the dataset.
        """
        # load coords, seqs, compute embs. that's all.
        data = {}
        decoy_pbd_files = [pdb.PDBFile.read(decoy_pdb_path) for decoy_pdb_path in decoy_pdb_paths]
        target_pbd_files = [pdb.PDBFile.read(target_pdb_path) for target_pdb_path in target_pdb_paths]
        data["decoy"] = [pdb.get_structure(decoy_pdb_file, model=1) for decoy_pdb_file in decoy_pbd_files]
        data["target"] = [pdb.get_structure(target_pdb_file, model=1) for target_pdb_file in target_pbd_files]
        data["decoy"] = [data["decoy"][data["decoy"].chain_id == decoy_chain_id] for decoy_chain_id in decoy_chain_ids]
        data["target"] = [data["target"][data["target"].chain_id == target_chain_id] for target_chain_id in target_chain_ids]
        return data

    @property
    def collate_fn(self):
        return partial(
            default_collate,
            max_len=self.crop_len,
            atom_idx=1,
        )


def align_chains(example: Dict):
    atom_tys = example["metadata"]["atom_tys"]
    bb_atom_posns = data_utils._backbone_atom_tensor(tuple(atom_tys))
    bb_atom_mask = torch.nn.functional.one_hot(bb_atom_posns, len(atom_tys))
    bb_atom_mask = torch.sum(bb_atom_mask, dim=0).bool()
    for i in range(len(example["decoy"]["coordinates"])):
        # align on common backbone atoms and residues
        decoy_atom_mask = example["decoy"]["atom_mask"][i]
        target_atom_mask = example["target"]["atom_mask"][i]
        decoy_residue_mask = example["decoy"]["residue_mask"][i]
        target_residue_mask = example["target"]["residue_mask"][i]
        alignment_residue_mask = decoy_residue_mask & target_residue_mask
        alignment_atom_mask = decoy_atom_mask & target_atom_mask
        # align onlly on valid backbone atoms
        alignment_atom_mask[..., ~bb_atom_mask] = False

        example["decoy"]["coordinates"][i] = fa_align(
            mobile=example["decoy"]["coordinates"][i],
            target=example["target"]["coordinates"][i],
            atom_mask=alignment_atom_mask,
            align_mask=alignment_residue_mask,
        )
    return example


def collate(batch: List[Optional[Dict]]) -> List[Dict]:
    return list(filter(exists, batch))


class ModelListDf:
    def __init__(
        self,
        df_path: str,
        decoy_pdb_folder: str,
        target_pdb_folder: str,
        split: str,
        use_bound: bool = False,
        monomer_ty: str = "holo",
        limit_clusters: int = -1,
        raise_exceptions: bool = True,
    ):
        self.target_pdb_folder = target_pdb_folder
        self.decoy_pdb_folder = decoy_pdb_folder
        self.use_bound = use_bound
        if df_path.endswith(".parquet"):
            index = pd.read_parquet(df_path)
        else:
            index = pd.read_csv(df_path)
        for col in index.select_dtypes(include=["category"]).columns:
            index[col] = index[col].astype("object")
        clusters = defaultdict(lambda: defaultdict(list))
        for i in range(len(index)):
            row = index.iloc[i].to_dict()
            cid = (
                row["cluster_id"] if row["split"] not in ["test", "invalid"] else str(i)
            )
            clusters[row["split"]][cid].append(row)
        clusterlists = []
        for i, cluster_id in enumerate(clusters[split]):
            clusterlists.append(clusters[split][cluster_id])
        self.clusters = self.filter_rows_by_monomer_ty(clusterlists, monomer_ty)
        if limit_clusters > 0:
            shuffle(self.clusters)
            self.clusters = self.clusters[:limit_clusters]
        self.monomer_ty = monomer_ty
        self.raise_exceptions = raise_exceptions

    def filter_rows_by_monomer_ty(self, clusterlists, monomer_ty):
        if monomer_ty == "all":
            return clusterlists
        filtered_rows = []
        for cluster in clusterlists:
            filtered_cluster = []
            for df_row in cluster:
                has_rec, has_lig = df_row[f"{monomer_ty}_R"], df_row[f"{monomer_ty}_L"]
            if has_lig and has_rec:
                filtered_cluster.append(df_row)
            if len(filtered_cluster) > 0:
                filtered_rows.append(filtered_cluster)
        return filtered_rows

    def restrict(self, start, end):
        self.clusters = self.clusters[start:end]
        return self

    def sample(self, index, split):
        df_row = split[index][np.random.choice(len(split[index]))]
        rec_pdb = df_row[f"{self.monomer_ty}_R_pdb"]
        lig_pdb = df_row[f"{self.monomer_ty}_L_pdb"]
        metadata = deepcopy(df_row)
        metadata["receptor_ty"] = self.monomer_ty
        metadata["ligand_ty"] = self.monomer_ty
        return rec_pdb, lig_pdb, metadata

    def __getitem__(self, idx):
        rec, lig, meta = self.sample(index=idx, split=self.clusters)
        tgt_fldr, decoy_fldr = self.target_pdb_folder, self.decoy_pdb_folder
        if "subfolder" in meta:
            tgt_fldr, decoy_fldr = map(
                lambda x: os.path.join(x, meta["subfolder"]), (tgt_fldr, decoy_fldr)
            )
        decoy_pdbs = list(map(lambda x: os.path.join(decoy_fldr, x), [rec, lig]))
        decoy_pdbs = list(
            map(lambda x: x if x.endswith("pdb") else f"{x}.pdb", decoy_pdbs)
        )
        target_pdbs = list(
            map(
                lambda x: os.path.join(tgt_fldr, x),
                [f"{meta['id']}.pdb", f"{meta['id']}.pdb"],
            )
        )
        decoy_chains, target_chains = ["R", "L"], ["R", "L"]
        exceptions = []
        for path in decoy_pdbs + target_pdbs:
            try:
                assert os.path.exists(path)
            except Exception as e:
                exceptions.append(e)
        if self.raise_exceptions and len(exceptions):
            print(f"failing with {len(exceptions)} exceptions")
            raise exceptions[0]
        return decoy_pdbs, target_pdbs, decoy_chains, target_chains, exceptions, meta

    def __len__(self):
        return len(self.clusters)
