import unittest
from pathlib import Path

from nuplan.common.utils.s3_utils import is_s3_path, split_s3_path


class TestS3Utils(unittest.TestCase):
    """
    A class to test that the S3 utilities for Sherlock function properly.
    """

    def test_is_s3_path(self) -> None:
        """
        Tests that the is_s3_path method works properly.
        """
        self.assertTrue(is_s3_path("s3://foo/bar/baz.txt"))
        self.assertFalse(is_s3_path("/foo/bar/baz"))
        self.assertFalse(is_s3_path("foo/bar/baz"))

    def test_split_s3_path(self) -> None:
        """
        Tests that the split_s3_path method works properly.
        """
        sample_s3_path = Path("s3://ml-caches/mitchell.spryn/foo/bar/baz.txt")
        expected_bucket = "ml-caches"
        expected_path = Path("mitchell.spryn/foo/bar/baz.txt")

        actual_bucket, actual_path = split_s3_path(sample_s3_path)

        self.assertEqual(expected_bucket, actual_bucket)
        self.assertEqual(expected_path, actual_path)


if __name__ == "__main__":
    unittest.main()
