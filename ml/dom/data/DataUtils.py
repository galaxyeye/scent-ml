from typing import Tuple

import numpy as np
from pyspark.mllib.linalg import Vectors, Vector
from pyspark.mllib.regression import LabeledPoint


class DataUtils:
    """
    Helper methods to load, save and pre-process data used in platon.ai format.
    """

    @staticmethod
    def parse_libsvm_line(line: str) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Parses a line in LIBSVM format into (label, indices, values).
        """
        items = line.split(None)
        label = float(items[0])
        nnz = len(items) - 1
        indices = np.zeros(nnz, dtype=np.int32)
        values = np.zeros(nnz)
        for i in range(nnz):
            index, value = items[1 + i].split(":")
            # 1-based to 0-based index
            indices[i] = int(index) - 1
            values[i] = float(value)
        return label, indices, values

    @staticmethod
    def parse_libsvm_line_to_labeled_point(size: int, line: str) -> LabeledPoint:
        (label, indices, values) = DataUtils.parse_libsvm_line(line)
        features = Vectors.sparse(size, indices, values)
        return LabeledPoint(label, features.toArray())

    @staticmethod
    def parse_libsvm_line_to_label_vector(size: int, line: str) -> (str, Vector):
        (label, indices, values) = DataUtils.parse_libsvm_line(line)
        features = Vectors.sparse(size, indices, values)
        return label, features

    @staticmethod
    def parse_libsvm_line_to_point(line: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parses a line in LIBSVM format into (label, indices, values).
        """
        items = line.split(None)
        nnz = len(items) - 1
        indices = np.zeros(nnz, dtype=np.int32)
        values = np.zeros(nnz)
        for i in range(nnz):
            index, value = items[i].split(":")
            # 1-based to 0-based index
            indices[i] = int(index) - 1
            values[i] = float(value)
        return indices, values
