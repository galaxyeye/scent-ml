import os
import sys

from ml.dom.data.DataUtils import DataUtils

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from unittest import TestCase

from pyspark.sql import SparkSession


class TestMLUtils(TestCase):

    def test__load_platon_ai_data(self):
        spark = SparkSession.builder.appName("DataLoadTest").getOrCreate()
        sc = spark.sparkContext
        sc.setLogLevel("WARN")

        path = "../../data/dom/amazon.dataset.libsvm.11.24.50.txt"
        lines = sc.textFile(path)

        print("=== rdd")
        rdd = lines.map(lambda line: line.split(" ^|^ ")).filter(lambda record: len(record) == 4)
        rdd.toDF().show(5)

        print("=== rdd2")
        rdd2 = rdd.map(lambda r: (
            r[0] + r[1], r[2], r[3]
        ))
        rdd2.toDF().show(5)

        print("=== rdd3")
        rdd3 = rdd2.map(lambda r: (
            DataUtils.parse_libsvm_line_to_labeled_point(r[0]), r[1], r[2]
        ))


        spark.stop()
