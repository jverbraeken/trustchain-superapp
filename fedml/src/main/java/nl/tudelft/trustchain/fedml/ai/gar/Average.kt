package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.concurrent.ConcurrentLinkedDeque

private val logger = KotlinLogging.logger("Average")

class Average : AggregationRule() {
    override fun integrateParameters(
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator,
        allOtherModelsBuffer: ArrayDeque<Pair<Int, INDArray>>,
        logging: Boolean,
        testBatches: List<DataSet?>,
        countPerPeer: Map<Int, Int>
    ): INDArray {
        logger.debug { formatName("Simple average") }
        val models = HashMap<Int, INDArray>()
        models[-1] = oldModel.sub(gradient)
        models.putAll(otherModels)
        logger.debug { "Found ${models.size} models in total" }
        val modelsAsArrays = models.map { it.value.toFloatMatrix()[0] }
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
