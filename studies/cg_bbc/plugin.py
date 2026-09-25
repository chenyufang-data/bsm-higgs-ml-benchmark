"""Local feature adapter; never deployed to the ROOT server."""

from hepml.adapters.object_inputs import role_kinematics

from .features import BASELINE_COLUMNS, DEFAULT_FEATURES, build_features

FEATURES = DEFAULT_FEATURES
RETAINED_COLUMNS = BASELINE_COLUMNS


def features_from_objects(objects, retained=False):
    """Model inputs (and, if retained, the baseline columns) from role four-vectors {"b1": (pt, eta, phi, mass), ...}."""
    return build_features(objects, label=0)[[*FEATURES, *RETAINED_COLUMNS] if retained else FEATURES]


def derive_features(frame):
    """Compute the model inputs and baseline columns from the saved selected-jet indices."""
    objects = {role: role_kinematics(frame, role, "jets") for role in ("b1", "b2", "c1")}
    derived = build_features(objects, label=0)
    result = frame.copy()
    for name in [*FEATURES, *RETAINED_COLUMNS]:
        result[name] = derived[name].to_numpy()
    return result
