package org.Spark.toeknee.mCNN

import breeze.linalg.{DenseMatrix => BDM}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

object mCNN {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    if (args.length < 3) {
      System.err.println("Usage: mCNN <training_set> <test_set> <maxIterations> <output>")
      System.exit(1)
    }

    val conf = new SparkConf().setAppName("CNN")
    //    val conf = new SparkConf().setAppName("CNN").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val trainlines = sc.textFile(args(0), 8)
    // Build data(classification, matrix),matrix is 28*28(0->783)
    val traindata = trainlines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => (arr(0), Vector2Tensor(Vectors.dense(arr.slice(1, 785).map(v => if (v > 200) 1.0 else 0)))))

    var i = 1
    var precision = 1.0
    val topology = new CNNTopology
    topology.addLayer(CNNLayer.buildConvolutionLayer(1, 6, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildMeanPoolingLayer(new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvolutionLayer(6, 12, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildMeanPoolingLayer(new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvolutionLayer(12, 12, new Scale(4, 4)))
    val cnn: CNN = new CNN(topology).setMaxIterations(args(2).toInt).setMiniBatchSize(16)

    val testlines = sc.textFile(args(1), 8)
    // Build data(classification, matrix),matrix is 28*28(0->783)
    val testdata = testlines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => (arr(0), Vector2Tensor(Vectors.dense(arr.slice(1, 785).map(v => if (v > 200) 1.0 else 0)))))

    val start = System.nanoTime()

    //    cnn.trainOneByOne(traindata)
    cnn.train(traindata, testdata)
    println("Training time: " + (System.nanoTime() - start) / 1e9)

    val right = testdata.map(record => {
      val result = cnn.predict(record._2)
      if (result == record._1) 1 else 0
    }).sum()
    precision = right.toDouble / testdata.count()
    println(s"Predicting precision: $right " + precision)

  }

  /**
    * set inlayer output
    * @param record
    */
  def Vector2Tensor(record: Vector): Array[BDM[Double]] = {
    val mapSize = new Scale(28, 28)
    val m = new BDM[Double](mapSize.x, mapSize.y)
    var i: Int = 0
    while (i < mapSize.x) {
      var j: Int = 0
      while (j < mapSize.y) {
        m(i, j) = record(mapSize.x * i + j)
        j += 1
      }
      i += 1
    }
    Array(m)
  }
}


