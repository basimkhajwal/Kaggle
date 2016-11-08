package logistic_regression

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.annotation.tailrec

/**
  * Created by Basim on 05/11/2016.
  */
class LogisticRegression(val trainData: DenseMatrix[Double], val results: DenseVector[Double]) {

  val m: Int = results.length

  def this(trainData: DenseMatrix[Double], results: DenseVector[Int])(implicit unused: DummyImplicit) {
    this(trainData, results.map(x => x * 1.0))
  }

  def sigmoid(x: Double): Double = 1.0 / (1 + Math.exp(-x))

  def hypothesis(theta: DenseMatrix[Double], mat: DenseMatrix[Double] = trainData): DenseVector[Double] =
    DenseVector((mat * theta).valuesIterator.map(sigmoid).toArray)

  def costGrad(theta: DenseMatrix[Double]): (Double, DenseMatrix[Double]) = {
    val hyp: DenseVector[Double] = hypothesis(theta)

    val a: Double = results.dot(hyp.map(Math.log))
    val b: Double = results.map(x => 1 - x).dot(hyp.map(x => Math.log(Math.max(1-x, 1e-50))))

    val err: DenseMatrix[Double] = (hyp - results).toDenseMatrix.t
    val grad: DenseMatrix[Double] = trainData.t * err

    ((a+b) * (-1.0 / m), grad * (1.0 / m))
  }

  @tailrec
  final def gradientDescent(theta: DenseMatrix[Double], alpha: Double, hist: DenseVector[Double], iterationsLeft: Int): (DenseMatrix[Double], DenseVector[Double]) = {
    val cost = costGrad(theta)

    if (iterationsLeft <= 0) (theta, hist)
    else gradientDescent(theta - (cost._2 * alpha), alpha, DenseVector.vertcat(hist, DenseVector(cost._1)), iterationsLeft - 1)
  }

  def solve( numIterations: Int = 400,
             alpha: Double = 0.001,
             initialTheta: DenseMatrix[Double] = DenseMatrix.zeros(trainData.cols, 1)
           ): (DenseMatrix[Double], DenseVector[Double]) = gradientDescent(initialTheta, alpha, DenseVector(), numIterations)
}
