package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.concurrent.ConcurrentLinkedDeque

private val logger = KotlinLogging.logger("Median")

fun trimmedMean(b: Int, l: List<Float>) = l.sorted().subList(b, l.size - b).average().toFloat()

class CWTrimmedMean(private val b: Int) : AggregationRule() {
    private val minimumModels = 2 * b + 1

    override fun integrateParameters(
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator,
        allOtherModelsBuffer: ConcurrentLinkedDeque<INDArray>
    ): INDArray {
        logger.debug { formatName("Coordinate-Wise Trimmed Mean") }
        val models = arrayListOf(oldModel.sub(gradient))
        otherModels.forEach { models.add(it) }
        logger.debug { "Found ${models.size} models in total" }
        return if (models.size < minimumModels) {
            models[0]
        } else {
            val modelsAsArrays = models.map { it.toFloatMatrix()[0] }
            val newMatrix = Array(1) { FloatArray(modelsAsArrays[0].size) }
            for (i in modelsAsArrays[0].indices) {
                val elements = ArrayList<Float>(modelsAsArrays.size)
                modelsAsArrays.forEach { elements.add(it[i]) }
                newMatrix[0][i] = trimmedMean(b, elements)
            }
            NDArray(newMatrix)
        }
    }

    override fun isDirectIntegration(): Boolean {
        return false
    }
}
