package logistic_regression

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by Basim on 05/11/2016.
  */
class LogisticRegression(val trainData: DenseMatrix[Double], val results: DenseVector[Double]) {

  def this(trainData: DenseMatrix[Double], results: DenseVector[Int])(implicit unused: DummyImplicit) {
    this(trainData, results.map(x => x * 1.0))
  }

  def sigmoid(x: Double): Double = 1.0 / (1 + Math.exp(-x))

  def hypothesis(theta: DenseMatrix[Double]): DenseVector[Double] = DenseVector((trainData * theta).valuesIterator.map(sigmoid).toArray)

  def costGrad(theta: DenseMatrix[Double]): (Double, DenseMatrix[Double]) = {
    val m: Int = theta.rows
    val hyp: DenseVector[Double] = hypothesis(theta)

    val a: Double = (-results).dot(hyp.map(Math.log))
    val b: Double = results.map(x => 1 - x).dot(hyp.map(x => Math.log(1-x)))

    val err: DenseMatrix[Double] = (hyp - results).toDenseMatrix.t
    val grad: DenseMatrix[Double] = trainData.t * err

    ((a-b) * (1.0 / m), grad * (1.0 / m))
  }

  def gradientDescent(theta: DenseMatrix[Double], iterationsLeft: Int): DenseMatrix[Double] = {
    if (iterationsLeft <= 0) theta
    else gradientDescent(theta - costGrad(theta)._2 * 0.1, iterationsLeft - 1)
  }

  def solve( numIterations: Int = 400,
             initialTheta: DenseMatrix[Double] = DenseMatrix.ones(trainData.cols, 1)
           ): DenseMatrix[Double] = gradientDescent(initialTheta, numIterations)
}
