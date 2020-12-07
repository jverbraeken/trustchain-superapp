package nl.tudelft.trustchain.fedml.ai


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.learning.GradientUpdater
import org.nd4j.linalg.learning.config.IUpdater
import org.nd4j.linalg.schedule.ISchedule
import java.util.*


data class EWCAdam constructor(
    private var learningRate: Double = DEFAULT_ADAM_LEARNING_RATE,
    private var learningRateSchedule: ISchedule? = null,
    internal val beta1: Double = DEFAULT_ADAM_BETA1_MEAN_DECAY,
    internal val beta2: Double = DEFAULT_ADAM_BETA2_VAR_DECAY,
    internal val epsilon: Double = DEFAULT_ADAM_EPSILON,
) : IUpdater {

    override fun stateSize(numParams: Long): Long {
        return 2 * numParams
    }

    override fun instantiate(viewArray: INDArray, initializeViewArray: Boolean): GradientUpdater<*> {
        val u = EWCAdamUpdater(this)
        var gradientShape = viewArray.shape()
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.size)
        gradientShape[1] = gradientShape[1] / 2
        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray)
        return u
    }

    override fun instantiate(updaterState: Map<String, INDArray>, initializeStateArrays: Boolean): GradientUpdater<*> {
        val u = EWCAdamUpdater(this)
        u.setState(updaterState, initializeStateArrays)
        return u
    }

    override fun clone(): EWCAdam {
        return EWCAdam(learningRate, learningRateSchedule, beta1, beta2, epsilon)
    }

    override fun getLearningRate(iteration: Int, epoch: Int): Double {
        return learningRateSchedule?.valueAt(iteration, epoch) ?: learningRate
    }

    override fun hasLearningRate(): Boolean {
        return true
    }

    override fun setLrAndSchedule(lr: Double, lrSchedule: ISchedule) {
        learningRate = lr
        learningRateSchedule = lrSchedule
    }

    companion object {
        const val DEFAULT_ADAM_LEARNING_RATE = 1e-3
        const val DEFAULT_ADAM_EPSILON = 1e-8
        const val DEFAULT_ADAM_BETA1_MEAN_DECAY = 0.9
        const val DEFAULT_ADAM_BETA2_VAR_DECAY = 0.999
    }
}
