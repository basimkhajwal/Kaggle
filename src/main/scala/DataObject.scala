
import breeze.linalg.{DenseMatrix, Matrix}

/**
  * Created by Basim on 05/11/2016.
  */
trait DataObject {

  def getData: Array[Double]
}

object DataObject {
  def getMatrix[U <: DataObject](data: Array[U]): Matrix[Double] = {
     new DenseMatrix[Double](
       data(0).getData.length + 1,
       data.length,
       data.foldLeft(Array.empty[Double])((arr, obj) => (arr :+ 1.0) ++ obj.getData)
     ).t // get the transpose
  }
}
