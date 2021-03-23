package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.d
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray

private val logger = KotlinLogging.logger("Bridge")

fun trimmedMean(b: Int, l: FloatArray): Float {
    l.sort()
    return l.copyOfRange(b, l.size - b).average().toFloat()
}

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
        logger.d(logging) { formatName("BRIDGE") }
        val models = HashMap<Int, INDArray>()
        val newModel = oldModel.sub(gradient)
        models[-1] = newModel
        models.putAll(newOtherModels)
        logger.d(logging) { "Found ${models.size} models in total" }
        return if (models.size < minimumModels) {
            newModel
        } else {
            val modelsAsArrays = models.map { toFloatArray(it.value) }.toTypedArray()
            val newVector = FloatArray(modelsAsArrays[0].size)
            for (i in modelsAsArrays[0].indices) {
                val elements = FloatArray(modelsAsArrays.size)
                modelsAsArrays.forEachIndexed { j, modelsAsArray -> elements[j] = modelsAsArray[i] }
                newVector[i] = trimmedMean(b, elements)
            }
            NDArray(Array(1) { newVector })
        }
    }

    private fun toFloatArray(first: INDArray): FloatArray {
        val data = first.data()
        val length = data.length().toInt()
        val indexer = data.indexer() as FloatRawIndexer
        val array = FloatArray(length)
        for (i in 0 until length) {
            array[i] = indexer.getRaw(i.toLong())
        }
        return array
    }

    override fun isDirectIntegration(): Boolean {
        return false
    }
}
