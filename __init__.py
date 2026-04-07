"""
PointNuker SOR — Statistical Outlier Removal plugin for LichtFeld Studio.
Extracted from PointNuker v1.0 (MIT) by the Radiance Fields community.
"""

import lichtfeld as lf

from .panels.sor_panel import SORPanel
from .panels.colmap_panel import COLMAPPanel
from .operators.sor_operator import SOROperator

_classes = [SORPanel, COLMAPPanel, SOROperator]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("pointnuker_sor loaded — SOR + COLMAP Tools ready.")


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("pointnuker_sor unloaded.")
