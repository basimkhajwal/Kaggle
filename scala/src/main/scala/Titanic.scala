import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import logistic_regression.LogisticRegression
import breeze.linalg._

/**
  * Created by Basim on 05/11/2016.
  */
object Titanic {

  val testData: String = "data/titanic/test.csv"
  val trainData: String = "data/titanic/train.csv"
  val outputFile: String = "data/titanic/predictionScaled.csv"

  case class PassengerData(id: Int, pClass: Int, isMale: Boolean, age: Double, sibSp: Int, parch: Int, fare: Double) extends DataObject {
    override def getFeatureVector(): Array[Double] =
      Array(if (isMale) 200 else 0, fare, pClass * 100, sibSp * 60, parch * 60)
  }

  def readPassenger(reader: CSVReader): PassengerData = {
    PassengerData(
      reader.getInt("PassengerId"),
      reader.getInt("Pclass"),
      reader.getString("Sex").equals("male"),
      if (reader.getString("Age").isEmpty) 0 else reader.getDouble("Age"),
      reader.getInt("SibSp"),
      reader.getInt("Parch"),
      if (reader.getString("Fare").isEmpty) 0 else reader.getDouble("Fare")
    )
  }

  def loadData(fileName: String): Array[PassengerData] = {
    new CSVReader(fileName).read(readPassenger).toArray
  }

  def loadDataResults(): Array[Boolean] = {
    new CSVReader(trainData).read(
      reader => {
        reader.getInt("Survived") == 1
      }
    ).toArray
  }

  def outputPredictions(logModel: LogisticRegression, theta: DenseMatrix[Double]): Unit = {
    val testData = loadData(Titanic.testData)
    val prediction = logModel.hypothesis(theta, DataObject.getDataMatrix(testData)).valuesIterator.map(x => if (x >= 0.5) 1 else 0).toArray

    val outputs: Array[(PassengerData, Int)] = testData.zip(prediction)

    val strData: String = "PassengerId,Survived\n" +
      outputs
        .map(res => res._1.id + "," + res._2 + "\n")
        .reduce(_+_)

    Files.write(Paths.get(outputFile), strData.getBytes(StandardCharsets.UTF_8))
  }

  def checkAccuracy(logModel: LogisticRegression, theta: DenseMatrix[Double]): Unit = {
    val cp = logModel.hypothesis(theta)
    val predictions = loadDataResults
    val totalCorrect = cp.valuesIterator.zip(predictions.iterator).filter(x => (x._1 >= 0.5) == x._2).length
    println("Training set correctly predicted: " + totalCorrect + " / " + cp.length)
  }

  def main(args: Array[String]): Unit = {

    val predictions = loadDataResults
    val trainData = loadData(this.trainData)

    val logModel = new LogisticRegression(DataObject.getDataMatrix(trainData), DenseVector(predictions.map(x => if (x) 1 else 0)))

    val before = System.currentTimeMillis
    val res = logModel.solve()
    println("Time taken: " + (System.currentTimeMillis - before) + "ms")
    println(res)

    checkAccuracy(logModel, res.toDenseMatrix.t)
  }

}
