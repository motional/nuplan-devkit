import unittest
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import PIL.Image as PilImg

from nuplan.database.utils.image import Image


class TestImageMock(unittest.TestCase):
    """Test suite for the Image class using mocks."""

    TEST_PATH = "nuplan.database.utils.image"

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self.mock_pil_img = MagicMock(PilImg.Image)
        self.image = Image(self.mock_pil_img)

    def test_as_pil(self) -> None:
        """Test the function as_pil."""
        img = self.image.as_pil
        self.assertEqual(self.mock_pil_img, img)

    @patch(f"{TEST_PATH}.np.array", autospec=True)
    def test_as_numpy_nocache(self, mock_array: Mock) -> None:
        """Test the function as_numpy_nocache."""
        _ = self.image.as_numpy_nocache()
        mock_array.assert_called_with(self.mock_pil_img, dtype=np.uint8)

    @patch(f"{TEST_PATH}.Image.as_numpy_nocache", autospec=True)
    def test_as_numpy(self, mock_as_numpy_nocache: Mock) -> None:
        """Test the function as_numpy_nocache."""
        _ = self.image.as_numpy
        mock_as_numpy_nocache.assert_called_once()

    @patch(f"{TEST_PATH}.cv2.cvtColor", autospec=True)
    @patch(f"{TEST_PATH}.np.array", autospec=True)
    def test_as_cv2_nocache(self, mock_array: Mock, mock_cvtcolor: Mock) -> None:
        """Test the function as_cv2_nocache."""
        _ = self.image.as_cv2_nocache()
        mock_cvtcolor.assert_called_with(mock_array(self.mock_pil_img, np.uint8), cv2.COLOR_RGB2BGR)

    @patch(f"{TEST_PATH}.Image.as_cv2_nocache", autospec=True)
    def test_as_cv2(self, mock_as_cv2_nocache: Mock) -> None:
        """Test the function as_numpy_nocache."""
        _ = self.image.as_cv2
        mock_as_cv2_nocache.assert_called_once()


class TestImage(unittest.TestCase):
    """Test suite for the Image class using synthetic image."""

    def setUp(self) -> None:
        """Inherited, see superclass"""
        pil_img: PilImg.Image = PilImg.new('RGB', (500, 500))
        self.image = Image(pil_img)

    def _test_numpy_type(self, img: Any) -> None:
        """
        Checks if the given object is a numpy array with dtype uint8.
        :param img: The image object to test. Type hint any because the test should be valid for all objects.
        """
        self.assertEqual(np.ndarray, type(img))
        self.assertEqual(np.uint8, img.dtype)
        self.assertNotEqual(np.float64, img.dtype)

    def test_as_pil(self) -> None:
        """Test the function as_pil."""
        img = self.image.as_pil
        self.assertEqual(PilImg.Image, type(img))

    def test_as_numpy_nocache(self) -> None:
        """Test the function as_numpy_nocache."""
        img = self.image.as_numpy_nocache()
        self._test_numpy_type(img)

    def test_as_numpy(self) -> None:
        """Test the function as_numpy_nocache."""
        img = self.image.as_numpy
        self._test_numpy_type(img)

    def test_as_cv2_nocache(self) -> None:
        """Test the function as_cv2_nocache."""
        img = self.image.as_cv2_nocache()
        self._test_numpy_type(img)

    def test_as_cv2(self) -> None:
        """Test the function as_numpy_nocache."""
        img = self.image.as_cv2
        self._test_numpy_type(img)
