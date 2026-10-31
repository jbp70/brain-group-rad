"""Displacement fields on the common processed grid: loading with an explicit convention,
warping, composition, Jacobian determinant.

Canonical convention (used everywhere inside this package):

- A field is an array [X, Y, Z, 3] of displacements in VOXELS of the common grid, components in
  grid-axis order (i, j, k). The processed grid (160x224x192, 1 mm, affine diag(+1, +1, +1)) is
  RAS-aligned, so "mm along world R/A/S" == "mm along grid axes i/j/k".
- fwd_i lives on the ATLAS grid and pulls subject i into atlas space:
      warped_i(x) = I_i(x + fwd_i(x)),  i.e. phi_i(x) = x + fwd_i(x) maps atlas -> subject coords.
- inv_i lives on SUBJECT i's grid and pulls the atlas into subject space:
      A_i(y) = A(y + inv_i(y)),        i.e. psi_i(y) = y + inv_i(y) maps subject -> atlas coords.
- Pair i -> j (subject i moved into subject j's space, on j's grid):
      u_ij(y) = inv_j(y) + fwd_i(y + inv_j(y))   (phi_i o psi_j).

A method's files are mapped onto this with a `Convention`: which axis each stored component is
(``components``, e.g. "zyx"), a sign per canonical axis (``signs``, e.g. "--+" for ITK/ANTs LPS
fields), units ("mm" or "vox"), and ``swap_roles`` (the method's "forward" file is our inv).
`evaluate check-convention` finds the convention empirically.
"""
import itertools
import os
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

AXES = "xyz"


class Convention:
    """How a method's stored displacement files map onto the canonical convention."""

    def __init__(self, components: str = "xyz", signs: str = "+++", units: str = "mm",
                 swap_roles: bool = False):
        if len(components) != 3 or sorted(components) != sorted(AXES):
            raise ValueError("components must be a permutation of 'xyz', got %r" % components)
        if len(signs) != 3 or any(s not in "+-" for s in signs):
            raise ValueError("signs must be 3 characters of '+'/'-', got %r" % signs)
        if units not in ("mm", "vox"):
            raise ValueError("units must be 'mm' or 'vox', got %r" % units)
        self.components = components
        self.signs = signs
        self.units = units
        self.swap_roles = bool(swap_roles)

    def describe(self) -> str:
        return "components=%s signs=%s units=%s swap_roles=%s" % (
            self.components, self.signs, self.units, self.swap_roles)

    def cli_flags(self) -> str:
        return "--components %s --signs %s --units %s%s" % (
            self.components, self.signs, self.units, " --swap-roles" if self.swap_roles else "")

    def as_dict(self) -> dict:
        return {"components": self.components, "signs": self.signs, "units": self.units,
                "swap_roles": self.swap_roles}

    def file_role(self, role: str) -> str:
        """Which of the method's files ('fwd' / 'inv') holds the canonical ``role`` field."""
        if role not in ("fwd", "inv"):
            raise ValueError(role)
        if not self.swap_roles:
            return role
        return "inv" if role == "fwd" else "fwd"

    def to_canonical(self, raw: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
        """raw [X, Y, Z, 3] in file component order -> canonical voxel displacements (float32).

        ``signs`` apply per canonical axis x, y, z (after reordering)."""
        out = np.empty(raw.shape, np.float32)
        for c, ax in enumerate(self.components):
            a = AXES.index(ax)
            scale = (-1.0 if self.signs[a] == "-" else 1.0)
            if self.units == "mm":
                scale /= float(spacing[a])
            np.multiply(raw[..., c], scale, out=out[..., a], casting="unsafe")
        return out


def all_conventions(units: str = "mm") -> List[Convention]:
    """Every component order x sign pattern x role assignment (6 x 8 x 2 = 96)."""
    out = []
    for comp in itertools.permutations(AXES):
        for signs in itertools.product("+-", repeat=3):
            for swap in (False, True):
                out.append(Convention("".join(comp), "".join(signs), units, swap))
    return out


# ------------------------------------------------------------------ loading
def load_array(path: str) -> np.ndarray:
    """Raw array from .nii/.nii.gz, .npy or .npz (single array, or key disp/field/flow/warp)."""
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        z = np.load(path)
        keys = list(z.keys())
        if len(keys) == 1:
            return z[keys[0]]
        for k in ("disp", "field", "flow", "warp"):
            if k in z:
                return z[k]
        raise ValueError("%s: several arrays %s and none named disp/field/flow/warp" % (path, keys))
    return np.asanyarray(nib.load(path).dataobj)


def to_xyz3(arr: np.ndarray, shape: Sequence[int], what: str = "field") -> np.ndarray:
    """[X,Y,Z,3], [X,Y,Z,1,3] (ITK NIfTI), [3,X,Y,Z] or with a leading batch 1 -> [X,Y,Z,3] float32."""
    shape = tuple(int(s) for s in shape)
    a = np.asarray(arr)
    a = a.reshape([d for d in a.shape if d != 1]) if a.ndim > 4 else a
    if a.ndim == 5 and a.shape[0] == 1:
        a = a[0]
    if a.shape == shape + (3,):
        return a.astype(np.float32, copy=False)
    if a.shape == (3,) + shape:
        return np.moveaxis(a, 0, -1).astype(np.float32)
    raise ValueError("%s has shape %s; expected %s or %s" % (what, a.shape, shape + (3,), (3,) + shape))


def load_raw_field(path: str, shape: Sequence[int]) -> np.ndarray:
    return to_xyz3(load_array(path), shape, path)


def load_field(path: str, shape: Sequence[int], spacing: Sequence[float], conv: Convention) -> np.ndarray:
    return conv.to_canonical(load_raw_field(path, shape), spacing)


class FieldSource:
    """Canonical fwd/inv fields per subject from path templates ({subject}, {subject_dir}), with
    a small LRU cache; ``identity=True`` returns zero fields (the no-registration reference)."""

    def __init__(self, fwd_template: Optional[str], inv_template: Optional[str], conv: Convention,
                 shape: Sequence[int], spacing: Sequence[float], identity: bool = False, cache: int = 16):
        self.templates = {"fwd": fwd_template, "inv": inv_template}
        self.conv = conv
        self.shape = tuple(shape)
        self.spacing = tuple(spacing)
        self.identity = identity
        self.capacity = max(int(cache), 2)
        self._cache: "OrderedDict[Tuple[str, str], np.ndarray]" = OrderedDict()
        self._preloaded = {}
        self._zero = None
        self.loads = 0
        if not identity and not (fwd_template and inv_template):
            raise ValueError("--fwd and --inv path templates are required (or use --identity)")

    def path(self, role: str, subject: dict) -> str:
        tpl = self.templates[self.conv.file_role(role)]
        return tpl.format(subject=subject["subject"], subject_dir=subject["dir"])

    def get(self, role: str, subject: dict) -> np.ndarray:
        if self.identity:
            if self._zero is None:
                self._zero = np.zeros(self.shape + (3,), np.float32)
                self._zero.flags.writeable = False
            return self._zero
        key = (role, subject["subject"])
        if key in self._preloaded:
            return self._preloaded[key]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        p = self.path(role, subject)
        if not os.path.isfile(p):
            raise FileNotFoundError("%s field of %s not found: %s" % (role, subject["subject"], p))
        f = load_field(p, self.shape, self.spacing, self.conv)
        f.flags.writeable = False
        self.loads += 1
        self._cache[key] = f
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return f

    def preload(self, role: str, subjects: Sequence[dict]) -> None:
        """Keep every ``role`` field in memory (shared copy-on-write by forked workers).

        ~83 MB per field on the 160x224x192 grid; pairs reuse fwd_i for every fixed subject j."""
        if self.identity:
            return
        for s in subjects:
            key = (role, s["subject"])
            if key not in self._preloaded:
                self._cache.pop(key, None)
                p = self.path(role, s)
                if not os.path.isfile(p):
                    raise FileNotFoundError("%s field of %s not found: %s" % (role, s["subject"], p))
                f = load_field(p, self.shape, self.spacing, self.conv)
                f.flags.writeable = False
                self.loads += 1
                self._preloaded[key] = f


# ------------------------------------------------------------------ warping
_GRIDS = {}


def identity_grid(shape: Sequence[int]) -> np.ndarray:
    """[3, X, Y, Z] float32 voxel coordinates (cached per shape)."""
    shape = tuple(int(s) for s in shape)
    if shape not in _GRIDS:
        _GRIDS.clear()
        g = np.indices(shape, dtype=np.float32)
        g.flags.writeable = False
        _GRIDS[shape] = g
    return _GRIDS[shape]


def _coords(disp: np.ndarray) -> np.ndarray:
    return identity_grid(disp.shape[:3]) + np.moveaxis(disp, -1, 0)


def warp(vol: np.ndarray, disp: np.ndarray, order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Pull ``vol`` through ``disp`` (canonical, defined on the output grid): out(x) = vol(x + disp(x)).

    order 1: trilinear (float32 output); order 0: nearest neighbour, keeps the dtype (labels)."""
    coords = _coords(disp)
    if order == 0:
        return ndimage.map_coordinates(vol, coords, order=0, mode="constant", cval=cval, output=vol.dtype)
    return ndimage.map_coordinates(vol.astype(np.float32, copy=False), coords, order=1, mode="constant",
                                   cval=cval, output=np.float32, prefilter=False)


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Displacement of (x -> x + a(x)) followed by b: u(x) = a(x) + b(x + a(x)), on a's grid.

    b is sampled trilinearly with edge clamping (mode 'nearest') outside its grid."""
    coords = _coords(a)
    out = np.array(a, np.float32, copy=True)
    for c in range(3):
        out[..., c] += ndimage.map_coordinates(b[..., c], coords, order=1, mode="nearest", output=np.float32,
                                               prefilter=False)
    return out


def jacobian_det(u: np.ndarray) -> np.ndarray:
    """det(I + grad u) of a canonical voxel displacement [X, Y, Z, 3] (np.gradient: central
    differences inside, one-sided at the border) -> [X, Y, Z] float32."""
    g = [[np.gradient(u[..., c], axis=a).astype(np.float32) for a in range(3)] for c in range(3)]
    for c in range(3):
        g[c][c] += 1.0
    return (g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1])
            - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0])
            + g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0]))


def norm_mm(u: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """|u| in mm of a canonical voxel displacement -> [X, Y, Z] float32."""
    s = np.asarray(spacing, np.float32)
    return np.sqrt(np.sum((u * s) ** 2, axis=-1))
