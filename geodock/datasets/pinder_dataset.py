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
from esm.inverse_folding.util import load_coords


def exists(x: Any) -> bool:
    """Returns True if and only if x is not None"""
    return x is not None


def default(x: Any, y: Any) -> Any:
    """Returns x if x exists, otherwise y"""
    return x if exists(x) else y


def identity(x):
    return x


class PinderDataset(torch.utils.data.Dataset):
    """Pinder Dataset, adapted copy from PDBDataset from Moleculearn. Only load the holo monomers pdbs"""

    def __init__(
        self,
        data_root_dir: str,
        raise_exceptions: bool,
        split: str,
        dataframe_path: Optional[str] = None,
        filter_fn: Optional[Any] = None,
        crop_len: int = -1,
        max_seq_len: int = -1,
    ):
        super().__init__()
        assert exists(dataframe_path)
        pdb_folder = os.path.join(data_root_dir, "pdbs")
        self.model_list = ModelListDf(
            pdb_folder=pdb_folder, df_path=dataframe_path, split=split
        )
        self.pdb_folder = pdb_folder
        self.filter_fn = default(filter_fn, identity)
        self.crop_len = crop_len
        self.max_seq_len = max_seq_len

    def subset(self, start: int, end: int):
        if isinstance(start, float) and end <= 1:
            start = int(start * len(self))
            end = int(end * len(self))
        self.model_list = self.model_list.restrict(start, end)
        return self

    def __len__(self):
        return len(self.model_list)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        item = None
        while item is None:
            (
                rec_pdb,
                lig_pdb,
                chain_ids,
                metadata,
            ) = self.model_list[idx]

            item = self.load_item(rec_pdb, lig_pdb, chain_ids, metadata)
            item = item if self.filter_fn(item) else None

            # load another example at random
            idx = randint(0, len(self) - 1)

        return item

    def load_item(
        self,
        rec_pdb: str,
        lig_pdb: str,
        chain_ids: List[str],
        metadata: dict,
    ) -> Dict:
        """
        Loads a single example from the dataset.
        """
        # load coords, seqs, compute embs
        data = {}
        rec_coords, rec_seq = load_coords(rec_pdb, chain_ids[0])
        lig_coords, lig_seq = load_coords(lig_pdb, chain_ids[1])
        data["receptor"] = {"coords": rec_coords, "seq": rec_seq}
        data["ligand"] = {"coords": lig_coords, "seq": lig_seq}

        # compute embeddings here? or preprocess?

        data["metadata"] = metadata
        return data


class ModelListDf:
    def __init__(
        self,
        pdb_folder: str,
        df_path: str,
        split: str,
    ):
        self.pdb_folder = pdb_folder
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
        self.clusters = self.filter_rows_by_monomer_ty(clusterlists, monomer_ty="holo")
        self.monomer_ty = "holo"

    def filter_rows_by_monomer_ty(self, clusterlists: List[List[Dict]], monomer_ty: str):
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

    def restrict(self, start: int, end: int):
        self.clusters = self.clusters[start:end]
        return self

    def sample(self, index: int, split: List[List[Dict]]):
        df_row = split[index][np.random.choice(len(split[index]))]
        rec_pdb = df_row[f"{self.monomer_ty}_R_pdb"]
        lig_pdb = df_row[f"{self.monomer_ty}_L_pdb"]
        metadata = deepcopy(df_row)
        metadata["receptor_ty"] = self.monomer_ty
        metadata["ligand_ty"] = self.monomer_ty
        return rec_pdb, lig_pdb, metadata

    def __getitem__(self, idx: int):
        rec, lig, meta = self.sample(index=idx, split=self.clusters)
        rec_pdb = os.path.join(self.pdb_folder, rec)
        lig_pdb = os.path.join(self.pdb_folder, lig)
        chains = ["R", "L"]
        exceptions = []
        for path in [rec_pdb, lig_pdb]:
            assert os.path.exists(path)
        return rec_pdb, lig_pdb, chains, meta

    def __len__(self):
        return len(self.clusters)
