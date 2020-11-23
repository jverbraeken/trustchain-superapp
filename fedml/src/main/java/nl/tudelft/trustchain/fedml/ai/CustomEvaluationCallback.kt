package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.nn.api.Model
import org.nd4j.evaluation.IEvaluation

interface CustomEvaluationCallback {
    fun call(listener: CustomEvaluativeListener, model: Model, invocationsCount: Long, evaluations: Array<out IEvaluation<*>>)
}
