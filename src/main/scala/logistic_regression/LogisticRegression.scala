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

  def costGradWithReg(theta: DenseMatrix[Double], lambda: Double): (Double, DenseMatrix[Double]) = {
    val (normCost, normGrad) = costGrad(theta)

    val thetaWithout0: DenseMatrix[Double] = DenseMatrix.vertcat[Double](DenseMatrix.ones(1, 1), DenseMatrix.zeros(theta.rows-1, 1)) :* theta

    val regCost: Double = (lambda / (2 * m)) * thetaWithout0.valuesIterator.map(x => x*x).sum
    val regGrad: DenseMatrix[Double] = (lambda / m) * thetaWithout0

    (normCost + regCost, normGrad + regGrad)
  }

  @tailrec
  final def gradientDescent(theta: DenseMatrix[Double], alpha: Double, lambda: Double, hist: DenseVector[Double], iterationsLeft: Int): (DenseMatrix[Double], DenseVector[Double]) = {
    val cost = costGradWithReg(theta, lambda)

    if (iterationsLeft <= 0) (theta, hist)
    else gradientDescent(theta - (cost._2 * alpha), alpha, lambda, DenseVector.vertcat(hist, DenseVector(cost._1)), iterationsLeft - 1)
  }

  def solve( numIterations: Int = 400,
             alpha: Double = 0.001,
             lambda: Double = 0,
             initialTheta: DenseMatrix[Double] = DenseMatrix.zeros(trainData.cols, 1)
           ): (DenseMatrix[Double], DenseVector[Double]) = gradientDescent(initialTheta, alpha, lambda, DenseVector(), numIterations)
}
