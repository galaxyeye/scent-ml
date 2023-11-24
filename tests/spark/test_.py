import os
import sys
import unittest

from pyspark.sql import SparkSession


class DataLoadCase(unittest.TestCase):
    def test_data_loading(self):
        spark = SparkSession \
            .builder \
            .appName("DataLoadTest") \
            .getOrCreate()
        sc = spark.sparkContext
        sc.setLogLevel("INFO")
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

        path = "../../data/dom/amazon.dataset.libsvm.11.24.50.txt"

        lines = sc.textFile(path)

        pairs = lines.map(lambda s: (s, 1))

        counts = pairs.reduceByKey(lambda a, b: a + b)


if __name__ == '__main__':
    unittest.main()
