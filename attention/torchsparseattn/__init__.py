from .fused import Fusedmax, FusedProxFunction
from .oscar import Oscarmax, OscarProxFunction
from .sparsemax import Sparsemax, SparsemaxFunction
from .fused_jv import _inplace_fused_prox_jv
from .isotonic import _inplace_contiguous_isotonic_regression, _make_unique
