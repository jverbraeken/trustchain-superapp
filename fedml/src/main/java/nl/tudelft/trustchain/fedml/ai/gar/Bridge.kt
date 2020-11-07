package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

private val logger = KotlinLogging.logger("Bridge")

class Bridge(val b: Int) : AggregationRule() {
    private val minimumModels = 2 * b + 1

    override fun integrateParameters(
        myModel: INDArray,
        gradient: INDArray,
        otherModelPairs: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): INDArray {
        logger.debug { formatName("BRIDGE") }
        val models: MutableList<INDArray> = arrayListOf(myModel)
        otherModelPairs.forEach { models.add(it) }
        logger.debug { "Found ${models.size} models in total" }
        return if (models.size < minimumModels) {
            myModel.sub(gradient)
        } else {
            val modelsAsArrays = models.map { it.toFloatMatrix()[0] }
            val newMatrix = Array(1) { FloatArray(modelsAsArrays[0].size) }
            for (i in modelsAsArrays[0].indices) {
                val elements = ArrayList<Float>(modelsAsArrays.size)
                modelsAsArrays.forEach { elements.add(it[i]) }
                newMatrix[0][i] = trimmedMean(b, elements)
            }
            NDArray(newMatrix).sub(gradient)
        }
    }
}
