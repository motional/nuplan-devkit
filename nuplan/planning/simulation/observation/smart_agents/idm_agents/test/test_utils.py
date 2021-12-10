import unittest

import numpy as np
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import transform_vector_global_to_local_frame, \
    transform_vector_local_to_global_frame


class IDMPolicyTests(unittest.TestCase):

    def setUp(self) -> None:
        self.test_vector = [1, 0, 0]

    def test_convert_global_to_local_frame(self):  # type: ignore
        result = transform_vector_global_to_local_frame(self.test_vector, np.pi / 2)
        expect = np.array([0, 1, 0])
        actual = np.array(result)
        self.assertTrue(np.allclose(expect, actual))

        result = transform_vector_global_to_local_frame(self.test_vector, -np.pi / 2)
        expect = np.array([0, -1, 0])
        actual = np.array(result)
        self.assertTrue(np.allclose(expect, actual))

    def test_convert_local_to_global_frame(self):  # type: ignore
        result = transform_vector_local_to_global_frame(self.test_vector, np.pi / 2)
        expect = np.array([0, -1, 0])
        actual = np.array(result)
        self.assertTrue(np.allclose(expect, actual))

        result = transform_vector_local_to_global_frame(self.test_vector, -np.pi / 2)
        expect = np.array([0, 1, 0])
        actual = np.array(result)
        self.assertTrue(np.allclose(expect, actual))

    if __name__ == '__main__':
        unittest.main()
