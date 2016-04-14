package org.Spark.toeknee.mCNN

import breeze.linalg.{DenseMatrix => BDM}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

object Example {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("CNN").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val trainlines = sc.textFile("./dataset/mnist_train_10000.csv", 8)
    // Build data(classification, matrix),matrix is 28*28(0->783)
    val traindata = trainlines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => (arr(0), Vector2Tensor(Vectors.dense(arr.slice(1, 785).map(v => if(v > 200) 1.0 else 0)))))

    val topology = new CNNTopology
    topology.addLayer(CNNLayer.buildConvolutionLayer(1, 6, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildMeanPoolingLayer(new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvolutionLayer(6, 12, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildMeanPoolingLayer(new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvolutionLayer(12, 12, new Scale(4, 4)))
    val cnn: CNN = new CNN(topology).setMaxIterations(5).setMiniBatchSize(16)
    val start = System.nanoTime()
    cnn.trainOneByOne(traindata)
    println("Training time: " + (System.nanoTime() - start) / 1e9)


    val testlines = sc.textFile("./dataset/mnist_test_1000.csv", 8)
    // Build data(classification, matrix),matrix is 28*28(0->783)
    val testdata = testlines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => (arr(0), Vector2Tensor(Vectors.dense(arr.slice(1, 785).map(v => if(v > 200) 1.0 else 0)))))

    val right = testdata.map(record =>{
      val result = cnn.predict(record._2)
      if(result == record._1) 1 else 0
    }).sum()
    println(s"Predicting precision: $right " + right.toDouble/(testdata.count()))

//    val testData = sc.textFile("dataset/mnist/mnist_test.csv", 8)
//      .map(line => line.split(",")).map(arr => arr.map(_.toDouble))
//      .map(arr => (arr(0), Example.Vector2Tensor(Vectors.dense(arr.slice(1, 785).map(v => if(v > 200) 1.0 else 0)))))
//
//    val rightM = data.map(record =>{
//      val result = cnn.predict(record._2)
//      if(result == record._1) 1 else 0
//    }).sum()
//    println(s"Mnist Full Predicting precision: $rightM " + rightM.toDouble/(data.count()))
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


