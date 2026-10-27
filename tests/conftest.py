"""
Fixtures communes : capteurs simulés à graine fixée.

La variable d'environnement EOQUAL_TEST_SEED décale toutes les graines
des fixtures, pour vérifier que les tolérances ne dépendent pas d'un
tirage favorable : EOQUAL_TEST_SEED=3 pytest
"""

import os

import pytest

from eoqual.measure.synthetic import MatrixSensor, SensorParams

SEED_OFFSET = int(os.environ.get("EOQUAL_TEST_SEED", "0"))


@pytest.fixture(scope="session")
def sensor() -> MatrixSensor:
    """Capteur S0 par défaut (256 × 256, 12 bits, K = 3 e⁻/DN)."""
    return MatrixSensor(SensorParams(seed=42 + SEED_OFFSET))


@pytest.fixture(scope="session")
def structured_sensor() -> MatrixSensor:
    """Capteur S0 avec DSNU et PRNU marquées sur les trois composantes."""
    return MatrixSensor(SensorParams(
        seed=7 + SEED_OFFSET,
        dsnu_pixel=1.0, dsnu_row=0.4, dsnu_col=0.8,
        prnu_pixel=0.01, prnu_row=0.003, prnu_col=0.005,
    ))
