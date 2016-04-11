package org.Spark.toeknee.mCNN

import java.io.Serializable

import breeze.linalg.{DenseMatrix => BDM}

abstract class CNNLayer private[mCNN] extends Serializable {

  def forward(input: Array[BDM[Double]]): Array[BDM[Double]] = input

  def prevDelta(nextDelta: Array[BDM[Double]], input: Array[BDM[Double]]): Array[BDM[Double]]

  def grad(delta: Array[BDM[Double]],
    layerInput: Array[BDM[Double]]): (Array[Array[BDM[Double]]], Array[Double]) = null
}

object CNNLayer {

  def buildConvolutionLayer(inMapNum: Int, outMapNum: Int, kernelSize: Scale): CNNLayer = {
    val layer = new ConvolutionLayer(inMapNum, outMapNum, kernelSize)
    layer
  }

  def buildMeanPoolingLayer(scaleSize: Scale): CNNLayer = {
    val layer = new MeanPoolingLayer(scaleSize)
    layer
  }

  def buildMaxPoolingLayer(scaleSize: Scale): CNNLayer = {
    val layer = new MaxPoolingLayer(scaleSize)
    layer
  }
}
