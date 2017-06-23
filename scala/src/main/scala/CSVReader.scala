import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by Basim on 22/05/2017.
  */
class CSVReader(val fileName: String) {

  private var headerMap: Map[String, Int] = Map.empty
  private var currentLine: Array[String] = Array.empty

  def getString(name: String): String = currentLine(headerMap(name.toLowerCase))

  def getDouble(name: String): Double = getString(name).toDouble

  def getInt(name: String): Int = getString(name).toInt

  def getColumns(line: String): Array[String] = {

    val columns = ArrayBuffer[String]()
    var currentCol = ""
    var inQuote = false

    for (c <- line) {
      if (c == '"') inQuote = !inQuote

      if (c == ',' && !inQuote) {
        columns.append(currentCol)
        currentCol = ""
      } else {
        currentCol += c
      }
    }

    columns.toArray
  }

  def read[T](readerFunc: CSVReader => T): List[T] = {
    val lines = Source.fromFile(fileName).getLines()

    val header = lines.next().toLowerCase()
    headerMap = getColumns(header).zipWithIndex.toMap

    var res: List[T] = Nil

    for (line <- lines) {
      currentLine = getColumns(line)
      res ::= readerFunc(this)
    }

    res
  }
}
