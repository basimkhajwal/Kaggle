import scala.io.Source

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
    val trainData = loadData(false)
    val predictions = loadPredictions
    val testData = loadData(true)

    println(DataObject.getMatrix(trainData))
  }

}
