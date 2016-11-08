import logistic_regression.LogisticRegression

import scala.io.Source
import breeze.linalg._
import breeze.plot._

/**
  * Created by Basim on 05/11/2016.
  */
object Titanic {

  val testData: String = "data/titanic/test.csv"
  val trainData: String = "data/titanic/train.csv"

  case class PassengerData(id: Int, pClass: Int, isMale: Boolean, age: Double, sibSp: Int, parch: Int, fare: Double) extends DataObject {
    override def getData: Array[Double] = Array(pClass, if (isMale) 0 else 1, age, sibSp, parch, fare)
  }

  def readColumns(isTest: Boolean, cols: Array[String]): PassengerData = {
    // Take into account extra comma in name
    val offset = if (isTest) 0 else 1
    PassengerData(
      cols(0).toInt,
      cols(1 + offset).toInt,
      cols(4 + offset) == "male",
      if (cols(5 + offset) == "") 0 else cols(5 + offset).toDouble,
      cols(6 + offset).toInt,
      cols(7 + offset).toInt,
      if (cols(9 + offset) == "") 0 else cols(9 + offset).toDouble
    )
  }

  def loadData(isTest: Boolean): Array[PassengerData] = {
    Source.fromFile(if (isTest) testData else trainData)
      .getLines().drop(1)
      .map(ln => readColumns(isTest, ln.split(","))).toArray
  }

  def loadPredictions(): Array[Boolean] = {
    Source.fromFile(trainData).getLines()
      .drop(1).map(ln => ln.split(",")(1) == "1").toArray
  }

  def main(args: Array[String]): Unit = {

    /*val trainData = new DenseMatrix[Double](4, 2, Array(1.0, 1, 1, 1, 3, 4, 8, 9))
    val results = DenseVector(Array(0, 0, 1, 1))
    val logRes = new LogisticRegression(trainData, results)

    val a = logRes.costGrad(new DenseMatrix[Double](2, 1, Array(-30.0, 6.0)))
    println("Cost: " + a._1)
    println("Grad: " + a._2)

    val optTheta = logRes.solve()
    println(logRes.hypothesis(optTheta, new DenseMatrix[Double](3, 2, Array(1.0, 1, 1, 2, 10, 7)) ))*/

    val trainData = loadData(false)
    val predictions = loadPredictions

    val logModel = new LogisticRegression(DataObject.getMatrix(trainData), DenseVector(predictions.map(x => if (x) 1 else 0)))
    val before: Long = System.currentTimeMillis();
    val res = logModel.solve(2000, 0.003)
    println("Time taken: " + (System.currentTimeMillis() - before) + "ms")

    val f = Figure("Cost over iterations")
    val p = f.subplot(0)
    val x: DenseVector[Double] = DenseVector((1 to 2000).toArray).map(x => 1.0 * x)
    p += plot(x, res._2, '.')
    p.xlabel = "Number of Iterations"
    p.ylabel = "Cost"

    val cp = logModel.hypothesis(res._1)
    val totalCorrect = cp.valuesIterator.zip(predictions.iterator).filter(x => (x._1 >= 0.5) == x._2).length
    println(totalCorrect + " / " + cp.length)
  }

}
