import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

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
  val outputFile: String = "data/titanic/predictionScaled.csv"

  case class PassengerData(id: Int, pClass: Int, isMale: Boolean, age: Double, sibSp: Int, parch: Int, fare: Double) extends DataObject {
    override def getData: Array[Double] =
      Array(if (isMale) 200 else 0, fare, pClass * 100, sibSp * 60, parch * 60)
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

  def loadDataResults(): Array[Boolean] = {
    Source.fromFile(trainData).getLines()
      .drop(1).map(ln => ln.split(",")(1) == "1").toArray
  }

  def outputPredictions(passengerData: Array[PassengerData], prediction: Iterable[Int]): Unit = {
    val strData: String = "PassengerId,Survived\n" +
      passengerData.zip(prediction).map(res => res._1.id + "," + res._2 + "\n").foldLeft("")(_+_)

    Files.write(Paths.get(outputFile), strData.getBytes(StandardCharsets.UTF_8))
  }

  def main(args: Array[String]): Unit = {

    val trainData = loadData(false)
    val predictions = loadDataResults

    println(DataObject.getMatrix(trainData))

    val numIterations = 25000

    val logModel = new LogisticRegression(DataObject.getMatrix(trainData), DenseVector(predictions.map(x => if (x) 1 else 0)))
    val before: Long = System.currentTimeMillis();
    val res = logModel.solve(numIterations, 0.0001)
    println("Time taken: " + (System.currentTimeMillis() - before) + "ms")

    val f = Figure("Cost over iterations")
    val p = f.subplot(0)
    val x: DenseVector[Double] = DenseVector((1 to numIterations).toArray).map(x => 1.0 * x)
    val y = res._2.map(_ * 100)
    p += plot(x, y, '.')
    p.xlabel = "Number of Iterations"
    p.ylabel = "Cost"

    for (i <- 1 to numIterations by 100) println("Iter " + i + ": " + res._2.valueAt(i))

    val cp = logModel.hypothesis(res._1)
    val totalCorrect = cp.valuesIterator.zip(predictions.iterator).filter(x => (x._1 >= 0.5) == x._2).length
    println("Training set correctly predicted: " + totalCorrect + " / " + cp.length)
    println(res._1)

    val testData = loadData(true)
    val predVals = logModel.hypothesis(res._1, DataObject.getMatrix(testData)).valuesIterator.map(x => if (x >= 0.5) 1 else 0)
    outputPredictions(testData, predVals.toIterable)
  }

}
