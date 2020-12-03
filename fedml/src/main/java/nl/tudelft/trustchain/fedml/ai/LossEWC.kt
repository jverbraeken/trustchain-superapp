package nl.tudelft.trustchain.fedml.ai

import mu.KotlinLogging
import org.nd4j.base.Preconditions
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.activations.impl.ActivationSoftmax
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.LossUtil
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.primitives.Pair

private val logger = KotlinLogging.logger("LossEWC")

/**
 * Multi-Class Cross Entropy loss function where each the output is (optionally) weighted/scaled by a fixed scalar value.
 * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
 * A weight vector of 1s should give identical results to no weight vector.
 *
 * @param weights Weights array (row vector). May be null.
 */
class LossEWC(private val softmaxClipEps: Double = DEFAULT_SOFTMAX_CLIPPING_EPSILON, private val weights: INDArray? = null) :
    ILossFunction {
    private val importance = 1000

    private fun scoreArray(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?): INDArray {
        var labels = labels
        if (!labels.equalShapes(preOutput)) {
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s",
                labels.shape(),
                preOutput.shape())
        }
        labels = labels.castTo(preOutput.dataType()) //No-op if already correct dtype
        val output = activationFn.getActivation(preOutput.dup(), true)
        if (activationFn is ActivationSoftmax && softmaxClipEps > 0.0) {
            BooleanIndexing.replaceWhere(output, softmaxClipEps, Conditions.lessThan(softmaxClipEps))
            BooleanIndexing.replaceWhere(output, 1.0 - softmaxClipEps, Conditions.greaterThan(1.0 - softmaxClipEps))
        }
        val scoreArr = Transforms.log(output, false).muli(labels)

        //Weighted loss function
        if (weights != null) {
            check(weights.length() == scoreArr.size(1)) {
                ("Weights vector (length ${weights.length()}) does not match output.size(1)=${preOutput.size(1)}")
            }
            scoreArr.muliRowVector(weights.castTo(scoreArr.dataType()))
        }
        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask)
        }
        return scoreArr
    }

    override fun computeScore(
        labels: INDArray,
        preOutput: INDArray,
        activationFn: IActivation,
        mask: INDArray?,
        average: Boolean,
    ): Double {
        val scoreArr = scoreArray(labels, preOutput, activationFn, mask)
        var score = -scoreArr.sumNumber().toDouble()
        if (average) {
            score /= scoreArr.size(0).toDouble()
        }
        return if (old_var_list != null && fishers != null) {
            val penalty = model.paramTable().map { (n, p) ->
                fishers!!.getValue(n).mul(p.sub(old_var_list!![n]).mul(p.sub(old_var_list!![n]))).sumNumber().toDouble()
            }.sum()
            logger.debug { "LossEWC: $score  <->  ${1000 * penalty}" }
            score + 1000 * penalty
        } else {
            score
        }
    }

    override fun computeScoreArray(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?): INDArray {
        val scoreArr = scoreArray(labels, preOutput, activationFn, mask)
        val a = scoreArr.sum(true, 1).muli(-1)
        val b = if (old_var_list != null && fishers != null) {
            val penalty = model.paramTable().map { (n, p) ->
                fishers!!.getValue(n).mul(p.sub(old_var_list!![n]).mul(p.sub(old_var_list!![n]))).sumNumber().toDouble()
            }.sum()
            20 * penalty
        } else {
            0
        }
        logger.debug { "LossEWC: $a ... $b" }
        return a.add(b)
    }

    override fun computeGradient(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?): INDArray {
        var labels = labels
        if (!labels.equalShapes(preOutput)) {
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s",
                labels.shape(),
                preOutput.shape())
        }
        val grad: INDArray
        val output = activationFn.getActivation(preOutput.dup(), true)
        labels = labels.castTo(preOutput.dataType()) //No-op if already correct dtype
        if (activationFn is ActivationSoftmax) {
            if (mask != null && LossUtil.isPerOutputMasking(output, mask)) {
                throw UnsupportedOperationException("Per output masking for MCXENT + softmax: not supported")
            }

            //Weighted loss function
            grad = if (weights != null) {
                check(weights.length() == output.size(1)) {
                    ("Weights vector (length " + weights.length()
                        + ") does not match output.size(1)=" + output.size(1))
                }
                val temp = labels.mulRowVector(weights.castTo(labels.dataType()))
                val col = temp.sum(true, 1)
                output.mulColumnVector(col).sub(temp)
            } else {
                output.subi(labels)
            }
        } else {
            val dLda = output.rdivi(labels).negi()
            grad = activationFn.backprop(preOutput, dLda).first

            //Weighted loss function
            if (weights != null) {
                check(weights.length() == output.size(1)) {
                    ("Weights vector (length " + weights.length()
                        + ") does not match output.size(1)=" + output.size(1))
                }
                grad.muliRowVector(weights.castTo(grad.dataType()))
            }
        }

        //Loss function with masking
        if (mask != null) {
            LossUtil.applyMask(grad, mask)
        }
        return grad
    }

    override fun computeGradientAndScore(
        labels: INDArray, preOutput: INDArray, activationFn: IActivation,
        mask: INDArray, average: Boolean,
    ): Pair<Double, INDArray> {
        return Pair(computeScore(labels, preOutput, activationFn, mask, average),
            computeGradient(labels, preOutput, activationFn, mask))
    }

    /**
     * The opName of this function
     *
     * @return
     */
    override fun name(): String {
        return toString()
    }

    override fun toString(): String {
        return if (weights == null) "LossEWC()" else "LossEWC(weights=$weights)"
    }

    companion object {
        var old_var_list: Map<String, INDArray>? = null
        var fishers: Map<String, INDArray>? = null
        lateinit var model: CustomMultiLayerNetwork
        var calculateFisher = false
        private const val DEFAULT_SOFTMAX_CLIPPING_EPSILON = 1e-10
    }
}
