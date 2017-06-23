
import breeze.linalg.DenseMatrix

/**
  * Created by Basim on 05/11/2016.
  */
trait DataObject {
  def getFeatureVector(): Array[Double]
}

object DataObject {
  def getDataMatrix[U <: DataObject](dataObjects: Array[U]): DenseMatrix[Double] = {
    val data = dataObjects map (_.getFeatureVector())
    val dataSize = data(0).length + 1
    val arrayData = new Array[Double](dataSize * data.length)

    var i = 0
    while (i < data.length) {
      arrayData(i * dataSize) = 1
      System.arraycopy(data(i), 0, arrayData, 1+i*dataSize, dataSize-1)
      i += 1
    }

    new DenseMatrix[Double](dataSize, data.length, arrayData).t
  }
}
