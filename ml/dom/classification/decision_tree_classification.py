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
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
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

    featureNames = "top-g0,top-g1,top-g2,top-g3,left-g0,left-g1,left-g2,left-g3,width-g0,width-g1,width-g2,width-g3,height-g0,height-g1,height-g2,height-g3,char-g0,char-g1,char-g2,char-g3,txt_nd-g0,txt_nd-g1,txt_nd-g2,txt_nd-g3,img-g0,img-g1,img-g2,img-g3,a-g0,a-g1,a-g2,a-g3,sibling-g0,sibling-g1,sibling-g2,sibling-g3,child-g0,child-g1,child-g2,child-g3,dep-g0,dep-g1,dep-g2,dep-g3,seq-g0,seq-g1,seq-g2,seq-g3,txt_dns-g0,txt_dns-g1,txt_dns-g2,txt_dns-g3,pid-g0,pid-g1,pid-g2,pid-g3,tag-g0,tag-g1,tag-g2,tag-g3,nd_id-g0,nd_id-g1,nd_id-g2,nd_id-g3,nd_cs-g0,nd_cs-g1,nd_cs-g2,nd_cs-g3,ft_sz-g0,ft_sz-g1,ft_sz-g2,ft_sz-g3,color-g0,color-g1,color-g2,color-g3,b_bolor-g0,b_bolor-g1,b_bolor-g2,b_bolor-g3,rtop-g0,rtop-g1,rtop-g2,rtop-g3,rleft-g0,rleft-g1,rleft-g2,rleft-g3,rrow-g0,rrow-g1,rrow-g2,rrow-g3,rcol-g0,rcol-g1,rcol-g2,rcol-g3,dist-g0,dist-g1,dist-g2,dist-g3,simg-g0,simg-g1,simg-g2,simg-g3,mimg-g0,mimg-g1,mimg-g2,mimg-g3,limg-g0,limg-g1,limg-g2,limg-g3,aimg-g0,aimg-g1,aimg-g2,aimg-g3,saimg-g0,saimg-g1,saimg-g2,saimg-g3,maimg-g0,maimg-g1,maimg-g2,maimg-g3,laimg-g0,laimg-g1,laimg-g2,laimg-g3,char_max-g0,char_max-g1,char_max-g2,char_max-g3,char_ave-g0,char_ave-g1,char_ave-g2,char_ave-g3,own_char-g0,own_char-g1,own_char-g2,own_char-g3,own_txt_nd-g0,own_txt_nd-g1,own_txt_nd-g2,own_txt_nd-g3,grant_child-g0,grant_child-g1,grant_child-g2,grant_child-g3,descend-g0,descend-g1,descend-g2,descend-g3,sep-g0,sep-g1,sep-g2,sep-g3,rseq-g0,rseq-g1,rseq-g2,rseq-g3,txt_nd_c-g0,txt_nd_c-g1,txt_nd_c-g2,txt_nd_c-g3,vcc-g0,vcc-g1,vcc-g2,vcc-g3,vcv-g0,vcv-g1,vcv-g2,vcv-g3,avcc-g0,avcc-g1,avcc-g2,avcc-g3,avcv-g0,avcv-g1,avcv-g2,avcv-g3,hcc-g0,hcc-g1,hcc-g2,hcc-g3,hcv-g0,hcv-g1,hcv-g2,hcv-g3,ahcc-g0,ahcc-g1,ahcc-g2,ahcc-g3,ahcv-g0,ahcv-g1,ahcv-g2,ahcv-g3,txt_df-g0,txt_df-g1,txt_df-g2,txt_df-g3,cap_df-g0,cap_df-g1,cap_df-g2,cap_df-g3,tn_max_w-g0,tn_max_w-g1,tn_max_w-g2,tn_max_w-g3,tn_ave_w-g0,tn_ave_w-g1,tn_ave_w-g2,tn_ave_w-g3,tn_max_h-g0,tn_max_h-g1,tn_max_h-g2,tn_max_h-g3,tn_ave_h-g0,tn_ave_h-g1,tn_ave_h-g2,tn_ave_h-g3,a_max_w-g0,a_max_w-g1,a_max_w-g2,a_max_w-g3,a_ave_w-g0,a_ave_w-g1,a_ave_w-g2,a_ave_w-g3,a_max_h-g0,a_max_h-g1,a_max_h-g2,a_max_h-g3,a_ave_h-g0,a_ave_h-g1,a_ave_h-g2,a_ave_h-g3,img_max_w-g0,img_max_w-g1,img_max_w-g2,img_max_w-g3,img_ave_w-g0,img_ave_w-g1,img_ave_w-g2,img_ave_w-g3,img_max_h-g0,img_max_h-g1,img_max_h-g2,img_max_h-g3,img_ave_h-g0,img_ave_h-g1,img_ave_h-g2,img_ave_h-g3,tn_total_w-g0,tn_total_w-g1,tn_total_w-g2,tn_total_w-g3,tn_total_h-g0,tn_total_h-g1,tn_total_h-g2,tn_total_h-g3,a_total_w-g0,a_total_w-g1,a_total_w-g2,a_total_w-g3,a_total_h-g0,a_total_h-g1,a_total_h-g2,a_total_h-g3,img_total_w-g0,img_total_w-g1,img_total_w-g2,img_total_w-g3,img_total_h-g0,img_total_h-g1,img_total_h-g2,img_total_h-g3".split(",")

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    # featureIndexer = VectorIndexer(inputCol="", outputCol="indexedFeatures", maxCategories=20).fit(data)
    vecAssembler = VectorAssembler(inputCols=featureNames, outputCol="features")

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, vecAssembler, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").where(predictions.indexedLabel != 0.0).show(100)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))

    treeModel = model.stages[2]
    # summary only
    print(treeModel)
    # $example off$

    spark.stop()
