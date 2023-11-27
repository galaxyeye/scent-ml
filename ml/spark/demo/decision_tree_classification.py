#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Decision Tree Classification Example.
"""
import os
import sys

# $example on$
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("DataLoader").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("INFO")
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    path = "C:\\Users\\pereg\\AppData\\Local\\Temp\\pulsar\\ml\\amazon.dataset.11.27.0-600.csv"
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(df.dtypes)

    spark.stop()
