import logging
from typing import TYPE_CHECKING

import numpy as np
from mpmath import whitw
from scipy.special import gamma

from ryd_numerov.radial.wavefunction import Wavefunction

if TYPE_CHECKING:
    from ryd_numerov.model import QuantumDefect
    from ryd_numerov.radial.grid import Grid
    from ryd_numerov.units import NDArray

logger = logging.getLogger(__name__)


class WavefunctionWhittaker(Wavefunction):
    def __init__(
        self,
        grid: "Grid",
        quantum_defect: "QuantumDefect",
    ) -> None:
        super().__init__(grid)
        self.quantum_defect = quantum_defect

    def integrate(self) -> "NDArray":
        nu = self.quantum_defect.n_star
        l = self.quantum_defect.l
        whitw_list = np.array([whitw(nu, l + 0.5, 2 * r / nu) for r in self.grid.x_list])

        if whitw_list[-1] == 0:
            logger.warning("Whittaker function is zero at the outer boundary, cannot determine correct sign.")
        elif np.sign(whitw_list[-1]) != (-1) ** (self.quantum_defect.n - self.quantum_defect.l - 1):
            whitw_list = -whitw_list

        u_list: NDArray = whitw_list / np.sqrt(nu * nu * gamma(nu + l + 1) * gamma(nu - l))
        w_list: NDArray = u_list / np.sqrt(self.grid.z_list)

        self._w_list = w_list
        return w_list
