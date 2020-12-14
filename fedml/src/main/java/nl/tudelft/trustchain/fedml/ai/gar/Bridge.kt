package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.concurrent.ConcurrentLinkedDeque

private val logger = KotlinLogging.logger("Bridge")

class Bridge(private val b: Int) : AggregationRule() {
    private val minimumModels = 2 * b + 1

    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
        logging: Boolean
    ): INDArray {
        debug(logging) { formatName("BRIDGE") }
        val models = HashMap<Int, INDArray>()
        models[-1] = oldModel
        models.putAll(newOtherModels)
        debug(logging) { "Found ${models.size} models in total" }
        return if (models.size < minimumModels) {
            oldModel.sub(gradient)
        } else {
            val modelsAsArrays = models.map { it.value.toFloatMatrix()[0] }
            val newMatrix = Array(1) { FloatArray(modelsAsArrays[0].size) }
            for (i in modelsAsArrays[0].indices) {
                val elements = ArrayList<Float>(modelsAsArrays.size)
                modelsAsArrays.forEach { elements.add(it[i]) }
                newMatrix[0][i] = trimmedMean(b, elements)
            }
            NDArray(newMatrix).sub(gradient)
        }
    }

    override fun isDirectIntegration(): Boolean {
        return false
    }
}
