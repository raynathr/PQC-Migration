"""
threshold.py

Derivation of CAS_min (Reviewer 1, comment 5).

The previous draft used CAS_min = 0.70 and labelled it illustrative. A
threshold that partitions organizations into assurance states should not
be a round number. Here it is derived from three stated inputs, any of
which an organization can substitute:

  r*      maximum tolerable exposure-weighted probability of break, i.e.
          the organization's risk appetite expressed on RQR_org;
  m_i^min minimum acceptable level for each non-risk CAS component;
  w       the CAS weight vector.

A posture is acceptable iff RQR_org(t) <= r* AND M_i(t) >= m_i^min for
every other component. The smallest CAS value consistent with that is

    CAS_min = w_AS (1 - r*) + sum_{i != AS} w_i m_i^min

which is a derived quantity, not a chosen one. It is a NECESSARY
condition: CAS >= CAS_min does not by itself certify every component
constraint, because CAS aggregates. We therefore report it as a
screening threshold and pair it with the component floors, rather than
presenting a single scalar as sufficient.

Anchors for r* and the floors:
  * CNSA 2.0 sets 2030-2033 transition deadlines for national-security
    systems, and NIST IR 8547 proposes deprecating 112-bit classical
    security by 2030 and disallowing it by 2035. An organization aligning
    to those deadlines is asserting that it intends residual exposure to
    be small, not merely declining, over the horizon.
  * The component floors are the levels below which the corresponding
    control is conventionally treated as failed rather than weak.
"""

import numpy as np

# Risk-appetite levels expressed on RQR_org. These are the elicitable
# quantity; the mapping from them to CAS_min is arithmetic.
RISK_APPETITE = {
    "stringent (national-security aligned, CNSA 2.0)": 0.05,
    "moderate (regulated enterprise)": 0.10,
    "permissive (general commercial)": 0.20,
}

COMPONENT_FLOORS = {
    "M_KM": 0.60,   # below this, key custody is failed, not weak
    "M_DC": 0.70,   # NIST IR 8547 deprecation posture by the horizon mid-point
    "M_CAI": 0.50,  # below this an organization cannot execute a swap at all
}


def cas_min(r_star, w, floors=None):
    floors = floors or COMPONENT_FLOORS
    return (w["AS"] * (1.0 - r_star)
            + w["KM"] * floors["M_KM"]
            + w["DC"] * floors["M_DC"]
            + w["CAI"] * floors["M_CAI"])


def table(w):
    return {k: round(cas_min(v, w), 4) for k, v in RISK_APPETITE.items()}


if __name__ == "__main__":
    w = {"AS": 0.40, "KM": 0.25, "DC": 0.20, "CAI": 0.15}
    for k, v in table(w).items():
        print(f"{k:48s} CAS_min = {v:.3f}")
