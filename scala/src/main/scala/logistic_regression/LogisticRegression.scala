package logistic_regression

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.optimize.{DiffFunction, LBFGS}

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
    (mat * theta).map(sigmoid).toDenseVector

  def costGrad(theta: DenseMatrix[Double]): (Double, DenseMatrix[Double]) = {
    val hyp: DenseVector[Double] = hypothesis(theta)

    val a: Double = results.dot(hyp.map(x => Math.log(Math.max(x, 1e-50))))
    val b: Double = results.map(x => 1 - x).dot(hyp.map(x => Math.log(Math.max(1-x, 1e-50))))

    val err: DenseMatrix[Double] = (hyp - results).toDenseMatrix
    val grad: DenseMatrix[Double] = (err * trainData).t

    ((a+b) * (-1.0 / m), grad * (1.0 / m))
  }

  def costGradWithReg(theta: DenseMatrix[Double], lambda: Double): (Double, DenseMatrix[Double]) = {
    val (normCost, normGrad) = costGrad(theta)

    val thetaWithout0 = theta.copy
    thetaWithout0.update(0, 0, 0)

    val regCost: Double = (lambda / (2 * m)) * thetaWithout0.valuesIterator.map(x => x*x).sum
    val regGrad: DenseMatrix[Double] = thetaWithout0 * (lambda / m)

    (normCost + regCost, normGrad + regGrad)
  }

  def solveIterative(
    numIterations: Int = 400,
    alpha: Double = 0.001,
    lambda: Double = 0,
    initialTheta: DenseMatrix[Double] = DenseMatrix.zeros(trainData.cols, 1)
  ): (DenseMatrix[Double], DenseVector[Double]) = {

    var i = 0
    var theta = initialTheta
    val hist = new DenseVector[Double](numIterations)

    while (i < numIterations) {
      val (cost, grad) = costGradWithReg(theta, lambda)
      theta = theta - grad * alpha
      hist(i) = cost
      i += 1
    }

    (theta, hist)
  }

  def solve(
    lambda: Double = 0, maxIterations: Int = -1,
    initialTheta: DenseVector[Double] = DenseVector.zeros(trainData.cols)
  ) : DenseVector[Double] = {

    val lbfgs = new LBFGS[DenseVector[Double]](maxIter=maxIterations)

    val diffFunc = new DiffFunction[DenseVector[Double]] {
      override def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val (c, err) = costGradWithReg(x.toDenseMatrix.t, lambda)
        (c, err.toDenseVector)
      }
    }

    lbfgs.minimize(diffFunc, initialTheta)
  }
}
