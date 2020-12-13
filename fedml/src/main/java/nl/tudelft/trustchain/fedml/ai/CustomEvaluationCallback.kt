package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.nn.api.Model
import org.nd4j.evaluation.IEvaluation

interface CustomEvaluationCallback {
    fun newSimulation(
        name: String,
        mlConfiguration: List<MLConfiguration>,
    )

    fun call(
        model: Model,
        evaluations: Array<out IEvaluation<*>>,
        simulationIndex: Int,
        score: Double,
        extraElements: Map<String, String>,
        elapsedTime: Long,
        iterations: Int,
        epoch: Int
    )
}
