package org.Spark.toeknee.mCNN

import breeze.linalg.{DenseMatrix => BDM}

class MaxPoolingLayer private[mCNN](scaleSize: Scale)  extends CNNLayer{

  def getScaleSize: Scale = scaleSize

  override def forward(input: Array[BDM[Double]]): Array[BDM[Double]] = {


    val lastMapNum: Int = input.length
    val output = new Array[BDM[Double]](lastMapNum)
    var i: Int = 0
    while (i < lastMapNum) {
      val lastMap: BDM[Double] = input(i)
      val scaleSize: Scale = this.getScaleSize
      output(i) =  maxPooling(lastMap, scaleSize) // MeanPoolingLayer.scaleMatrix(lastMap, scaleSize)
      i += 1
    }
    output
  }

  override def prevDelta(nextDelta: Array[BDM[Double]],
                         layerInput: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layerInput.length
    val errors = new Array[BDM[Double]](mapNum)
    var m: Int = 0
    val scale: Scale = this.getScaleSize
    while (m < mapNum) {
      val nextError: BDM[Double] = nextDelta(m)
      val map: BDM[Double] = layerInput(m)
      var outMatrix: BDM[Double] = (1.0 - map)
      outMatrix = map :* outMatrix
      val backMaxMatrix = backFilling(map, nextError)
      outMatrix = outMatrix :* backMaxMatrix
      errors(m) = outMatrix
      m += 1
    }
    errors
  }

  private[mCNN] def maxPooling(matrix: BDM[Double], scale: Scale): BDM[Double] = {
    val m: Int = matrix.rows
    val n: Int = matrix.cols
    val scaleX = scale.x
    val scaleY = scale.y
    val sm: Int = m / scaleX
    val sn: Int = n / scaleY
    val outMatrix = new BDM[Double](sm, sn)

    var i = 0  // iterate through blocks
    while (i < sm) {
      var j = 0
      while (j < sn) {
        var max = matrix(i * scaleX, j * scaleY) // initial to left up corner of the block
        var bi = i * scaleX // block i
        val biMax = (i + 1) * scaleX
        val bjMax = (j + 1) * scaleY
        while (bi < biMax) {
          var bj = j * scaleY // block j
          while (bj < bjMax) {
            max = if(matrix(bi, bj) > max) matrix(bi, bj) else max
            bj += 1
          }
          bi += 1
        }
        outMatrix(i, j) = max
        j += 1
      }
      i += 1
    }
    outMatrix
  }

  private[mCNN] def backFilling(inputMap: BDM[Double], nextError: BDM[Double]): BDM[Double] ={
    val scale = this.getScaleSize
    val backMaxMatrix = BDM.zeros[Double](inputMap.rows, inputMap.cols)
    val scaleX = scale.x
    val scaleY = scale.y
    val sm: Int = inputMap.rows / scaleX
    val sn: Int = inputMap.cols / scaleY
    var i = 0  // iterate through blocks
    while (i < sm) {
      var j = 0
      while (j < sn) {
        var max = inputMap(i * scaleX, j * scaleY) // initial to left up corner of the block
        var maxi = i * scaleX
        var maxj = j * scaleY
        var bi = i * scaleX // block i
        val biMax = (i + 1) * scaleX
        val bjMax = (j + 1) * scaleY
        while (bi < biMax) {
          var bj = j * scaleY // block j
          while (bj < bjMax) {
            if(inputMap(bi, bj) > max) {
              max = inputMap(bi, bj)
              maxi = bi
              maxj = bj
            }
            bj += 1
          }
          bi += 1
        }
        backMaxMatrix(maxi, maxj) = nextError(i, j) * scaleX * scaleY
        j += 1
      }
      i += 1
    }
    backMaxMatrix
  }

}
