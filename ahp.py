"""
ahp.py

AHP representability check for the CAS weight vector (Editor comment 5,
Reviewer 1 comment 2).

We cannot report an executed expert panel, and we do not claim one. What
we can do is show that the paper's stated HNDL-motivated weight vector is
REPRESENTABLE as a consistent set of Saaty-scale pairwise judgements --
i.e. that it is the kind of object an AHP elicitation returns, and that
the elicitation instrument in Appendix E would be able to reproduce or
refute it. A weight vector that could only arise from an inconsistent
judgement matrix would not survive elicitation at all.

Judgement matrix (Saaty 1-9 scale), stated as the paper's own reasoning:
  AS vs KM  = 2   algorithm strength moderately favoured over key mgmt
  AS vs DC  = 2   ... over deployment coverage
  AS vs CAI = 3   ... moderately-to-strongly over crypto-agility
  KM vs DC  = 1   key management and coverage judged equal
  KM vs CAI = 2   key management moderately over agility
  DC vs CAI = 1   coverage and agility judged equal
"""
import numpy as np

LABELS = ["AS", "KM", "DC", "CAI"]
J = np.array([
    [1.0, 2.0, 2.0, 3.0],
    [0.5, 1.0, 1.0, 2.0],
    [0.5, 1.0, 1.0, 1.0],
    [1 / 3, 0.5, 1.0, 1.0],
])
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12}  # Saaty random index


def ahp(J):
    n = J.shape[0]
    vals, vecs = np.linalg.eig(J)
    i = int(np.argmax(vals.real))
    w = np.abs(vecs[:, i].real); w = w / w.sum()
    lmax = vals[i].real
    ci = (lmax - n) / (n - 1)
    cr = ci / RI[n]
    return w, lmax, ci, cr


if __name__ == "__main__":
    w, lmax, ci, cr = ahp(J)
    stated = np.array([0.40, 0.25, 0.20, 0.15])
    print("AHP principal eigenvector:", dict(zip(LABELS, np.round(w, 4))))
    print("stated paper weights     :", dict(zip(LABELS, stated)))
    print("max abs deviation        : %.4f" % np.abs(w - stated).max())
    print("lambda_max=%.4f  CI=%.4f  CR=%.4f  (consistent if CR < 0.10)" % (lmax, ci, cr))
