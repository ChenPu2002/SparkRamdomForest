// RandomForestDemo.scala
package com.example

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.SparkSession

object RandomForestDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("RandomForestExample")
      .master("spark://spark-master:7077")
      .getOrCreate()

    // 读取数据
    val data = spark.read.format("libsvm").load("file:///data/sample_libsvm_data.txt")

    // 拆分数据集
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2))

    // 训练模型
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val model = rf.fit(train)

    // 预测并评估（示例仅输出部分结果）
    val predictions = model.transform(test)
    predictions.select("label", "prediction").show(5)

    spark.stop()
  }
}