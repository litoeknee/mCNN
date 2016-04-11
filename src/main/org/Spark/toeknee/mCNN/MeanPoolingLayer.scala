package org.Spark.toeknee.mCNN

import breeze.linalg.{DenseMatrix => BDM, kron}

class MeanPoolingLayer private[mCNN](scaleSize: Scale)
  extends CNNLayer{

  def getScaleSize: Scale = scaleSize

  override def forward(input: Array[BDM[Double]]): Array[BDM[Double]] = {
    val lastMapNum: Int = input.length
    val output = new Array[BDM[Double]](lastMapNum)
    var i: Int = 0
    while (i < lastMapNum) {
      val lastMap: BDM[Double] = input(i)
      val scaleSize: Scale = this.getScaleSize
      output(i) = MeanPoolingLayer.scaleMatrix(lastMap, scaleSize)
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
      outMatrix = outMatrix :* MeanPoolingLayer.kronecker(nextError, scale)
      errors(m) = outMatrix
      m += 1
    }
    errors
  }

}

object MeanPoolingLayer{

  private[mCNN] def kronecker(matrix: BDM[Double], scale: Scale): BDM[Double] = {
    val ones = BDM.ones[Double](scale.x, scale.y)
    kron(matrix, ones)
  }

  /**
   * return a new matrix that has been scaled down
   *
   * @param matrix
   */
  private[mCNN] def scaleMatrix(matrix: BDM[Double], scale: Scale): BDM[Double] = {
    val m: Int = matrix.rows
    val n: Int = matrix.cols
    val sm: Int = m / scale.x
    val sn: Int = n / scale.y
    val outMatrix = new BDM[Double](sm, sn)
    val size = scale.x * scale.y
    var i = 0
    while (i < sm) {
      var j = 0
      while (j < sn) {
        var sum = 0.0
        var si = i * scale.x
        while (si < (i + 1) * scale.x) {
          var sj = j * scale.y
          while (sj < (j + 1) * scale.y) {
            sum += matrix(si, sj)
            sj += 1
          }
          si += 1
        }
        outMatrix(i, j) = sum / size
        j += 1
      }
      i += 1
    }
    outMatrix
  }
}
