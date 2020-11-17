package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.concurrent.ConcurrentLinkedDeque

private val logger = KotlinLogging.logger("Average")

class Average : AggregationRule() {
    override fun integrateParameters(
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator,
        allOtherModelsBuffer: ConcurrentLinkedDeque<INDArray>,
        logging: Boolean
    ): INDArray {
        logger.debug { formatName("Simple average") }
        val models = arrayListOf(oldModel.sub(gradient))
        otherModels.forEach { models.add(it) }
        logger.debug { "Found ${models.size} models in total" }
        val modelsAsArrays = models.map { it.toFloatMatrix()[0] }
        val newMatrix = Array(1) { FloatArray(modelsAsArrays[0].size) }
        for (i in modelsAsArrays[0].indices) {
            val elements = ArrayList<Float>(modelsAsArrays.size)
            modelsAsArrays.forEach { elements.add(it[i]) }
            newMatrix[0][i] = elements.average().toFloat()
        }
        return NDArray(newMatrix)
    }

    override fun isDirectIntegration(): Boolean {
        return false
    }
}
