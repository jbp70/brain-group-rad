"""Fixed, dataset-independent label lookup tables for `export-npy`.

Each LUT maps original label ids to contiguous ids 0..K-1 (background 0 stays
0), so label maps fit uint8 when K <= 256. The table depends only on the label
set name (the ``labels_<name>.nii.gz`` file name), never on the data, so every
subject and dataset exported with the same name gets identical ids. Any value
missing from the LUT is an error (no silent assignment).

  synthseg  SynthSeg 2.0 with --parc: the 32 structures of SynthSeg's
            "labels table.txt" (FreeSurfer ids, incl. 24 CSF; 3/42 cortex kept
            in case a voxel is not parcellated) + Desikan-Killiany cortical
            parcels 1001-1035 / 2001-2035 (1004/2004 corpus callosum is listed
            but SynthSeg does not output it). 103 ids -> uint8.
  train18   training label set (many-to-one merge of the raw SynthSeg --parc
            ids): 18 foreground classes + background, left/right merged, all
            cortex (3/42 and DK parcels 1001-1035/2001-2035) merged into one
            class. Fixed ids (TRAIN18_CLASSES):
               0 background (0)                  10 pallidum (13, 52)
               1 cerebral white matter (2, 41)   11 3rd ventricle (14)
               2 cerebral cortex (3, 42, 1001-   12 4th ventricle (15)
                 1035, 2001-2035)                13 brainstem (16)
               3 lateral ventricle (4, 43)       14 hippocampus (17, 53)
               4 inferior lateral ventricle      15 amygdala (18, 54)
                 (5, 44)                         16 CSF (24)
               5 cerebellum white matter (7, 46) 17 accumbens (26, 58)
               6 cerebellum cortex (8, 47)       18 ventral DC (28, 60)
               7 thalamus (10, 49)
               8 caudate (11, 50)
               9 putamen (12, 51)
            Applied to labels_synthseg.nii.gz (raw SynthSeg ids) -> uint8.
  lumir10   LUMIR25 coarse validation labels: assumed 0..10 (10 classes +
            background; values UNVERIFIED -- export fails loudly if wrong).
  seg4      Neurite-OASIS seg4: assumed 0..4 (UNVERIFIED).
  seg35     Neurite-OASIS seg35: assumed 0..35 (UNVERIFIED).

Example:
    from preprocessing.luts import get_lut, apply_lut
    lut = get_lut("synthseg")
    small = apply_lut(seg, lut, "uint8")
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

SYNTHSEG_STRUCTURES: List[Tuple[int, str]] = [
    (0, "background"), (2, "left cerebral white matter"), (3, "left cerebral cortex"),
    (4, "left lateral ventricle"), (5, "left inferior lateral ventricle"),
    (7, "left cerebellum white matter"), (8, "left cerebellum cortex"), (10, "left thalamus"),
    (11, "left caudate"), (12, "left putamen"), (13, "left pallidum"), (14, "3rd ventricle"),
    (15, "4th ventricle"), (16, "brain-stem"), (17, "left hippocampus"), (18, "left amygdala"),
    (24, "CSF"), (26, "left accumbens area"), (28, "left ventral DC"),
    (41, "right cerebral white matter"), (42, "right cerebral cortex"), (43, "right lateral ventricle"),
    (44, "right inferior lateral ventricle"), (46, "right cerebellum white matter"),
    (47, "right cerebellum cortex"), (49, "right thalamus"), (50, "right caudate"), (51, "right putamen"),
    (52, "right pallidum"), (53, "right hippocampus"), (54, "right amygdala"),
    (58, "right accumbens area"), (60, "right ventral DC"),
]

DK_PARCELS = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "corpuscallosum", "cuneus", "entorhinal",
    "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital",
    "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
    "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral",
    "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
    "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]  # FreeSurfer ids 1001..1035 (lh) and 2001..2035 (rh), in this order


def _synthseg_entries() -> List[Tuple[int, str]]:
    out = list(SYNTHSEG_STRUCTURES)
    for base, hemi in ((1000, "lh"), (2000, "rh")):
        out += [(base + i + 1, "ctx-%s-%s" % (hemi, n)) for i, n in enumerate(DK_PARCELS)]
    return out


def _identity(n: int) -> List[Tuple[int, str]]:
    return [(i, "background" if i == 0 else "label %d" % i) for i in range(n + 1)]


def _build(entries: List[Tuple[int, str]], source: str) -> Dict[str, object]:
    ids = [i for i, _ in entries]
    assert ids[0] == 0 and len(set(ids)) == len(ids)
    mapping = OrderedDict((int(o), k) for k, o in enumerate(ids))
    names = OrderedDict((int(o), n) for o, n in entries)
    return {"map": mapping, "names": names, "source": source}


_DK_LH = list(range(1001, 1036))
_DK_RH = list(range(2001, 2036))

# (new id, name, raw SynthSeg ids); ids and names are fixed (design doc 3.4).
TRAIN18_CLASSES: List[Tuple[int, str, List[int]]] = [
    (0, "background", [0]),
    (1, "cerebral white matter", [2, 41]),
    (2, "cerebral cortex", [3, 42] + _DK_LH + _DK_RH),
    (3, "lateral ventricle", [4, 43]),
    (4, "inferior lateral ventricle", [5, 44]),
    (5, "cerebellum white matter", [7, 46]),
    (6, "cerebellum cortex", [8, 47]),
    (7, "thalamus", [10, 49]),
    (8, "caudate", [11, 50]),
    (9, "putamen", [12, 51]),
    (10, "pallidum", [13, 52]),
    (11, "3rd ventricle", [14]),
    (12, "4th ventricle", [15]),
    (13, "brainstem", [16]),
    (14, "hippocampus", [17, 53]),
    (15, "amygdala", [18, 54]),
    (16, "CSF", [24]),
    (17, "accumbens", [26, 58]),
    (18, "ventral DC", [28, 60]),
]


def _build_merged(classes: List[Tuple[int, str, List[int]]], source: str) -> Dict[str, object]:
    """Many-to-one LUT: every raw id of a class maps to the class id."""
    assert [k for k, _, _ in classes] == list(range(len(classes)))
    mapping: "OrderedDict[int, int]" = OrderedDict()
    names: "OrderedDict[int, str]" = OrderedDict()
    for k, name, ids in classes:
        for o in ids:
            assert o not in mapping, o
            mapping[int(o)] = k
            names[int(o)] = name
    return {"map": mapping, "names": names, "source": source,
            "classes": OrderedDict((k, n) for k, n, _ in classes)}


LUTS: Dict[str, Dict[str, object]] = {
    "synthseg": _build(_synthseg_entries(), "SynthSeg 2.0 labels table + DK parcellation (--parc)"),
    "train18": _build_merged(TRAIN18_CLASSES, "SynthSeg 2.0 --parc ids merged to 18 training classes "
                                              "(left/right and all cortex merged)"),
    "lumir10": _build(_identity(10), "LUMIR25 coarse labels, assumed 0..10 (unverified)"),
    "seg4": _build(_identity(4), "Neurite-OASIS seg4, assumed 0..4 (unverified)"),
    "seg35": _build(_identity(35), "Neurite-OASIS seg35, assumed 0..35 (unverified)"),
}


def get_lut(name: str, luts: Optional[Dict[str, Dict[str, object]]] = None) -> Optional[Dict[str, object]]:
    return (LUTS if luts is None else luts).get(name)


def lut_dtype(lut: Dict[str, object]) -> str:
    """uint8 when the remapped ids fit, else uint16."""
    return "uint8" if n_classes(lut) <= 256 else "uint16"


def n_classes(lut: Dict[str, object]) -> int:
    """Number of output ids (max new id + 1; == len(map) for one-to-one LUTs)."""
    return int(max(lut["map"].values())) + 1  # type: ignore[attr-defined]


def apply_lut(arr: np.ndarray, lut: Dict[str, object], dtype: str, what: str = "labels") -> np.ndarray:
    """Remap label ids with ``lut``; ValueError listing any value not in the LUT."""
    mapping: Dict[int, int] = lut["map"]  # type: ignore[assignment]
    r = np.rint(np.asarray(arr, dtype=np.float64))
    if r.size and r.min() < 0:
        raise ValueError("%s: negative label values" % what)
    vals = np.unique(r).astype(np.int64)
    unknown = [int(v) for v in vals if int(v) not in mapping]
    if unknown:
        raise ValueError("%s: label values %s are not in the fixed LUT (%s); export with "
                         "--no-remap-labels or extend preprocessing/luts.py" % (what, unknown[:20], lut["source"]))
    table = np.zeros(int(max(mapping)) + 1, dtype=np.int64)
    for o, k in mapping.items():
        table[o] = k
    out = table[r.astype(np.int64)]
    info = np.iinfo(np.dtype(dtype))
    if out.size and out.max() > info.max:
        raise ValueError("%s: remapped ids up to %d do not fit %s" % (what, out.max(), dtype))
    return out.astype(dtype)


def lut_json(lut: Dict[str, object]) -> Dict[str, object]:
    """JSON-friendly form: {"source", "dtype", "n_labels", "map": {orig: new}, "names": {orig: name}}.

    n_labels is the number of output ids. Merged LUTs (train18) also get
    "classes": {new id: class name}.
    """
    mapping: Dict[int, int] = lut["map"]  # type: ignore[assignment]
    names: Dict[int, str] = lut.get("names") or {}  # type: ignore[assignment]
    out = {"source": lut.get("source", ""), "dtype": lut_dtype(lut), "n_labels": n_classes(lut),
           "map": OrderedDict((str(o), int(k)) for o, k in mapping.items()),
           "names": OrderedDict((str(o), n) for o, n in names.items())}
    if lut.get("classes"):
        out["classes"] = OrderedDict((str(k), n) for k, n in lut["classes"].items())  # type: ignore[attr-defined]
    return out
