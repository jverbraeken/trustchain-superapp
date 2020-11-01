package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

private val logger = KotlinLogging.logger("Median")

fun median(l: List<Double>) = l.sorted().let { (it[it.size / 2] + it[(it.size - 1) / 2]) / 2 }

class Median : AggregationRule() {

    override fun integrateParameters(
        myModel: Pair<INDArray, Int>,
        otherModelPairs: List<Pair<INDArray, Int>>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): Pair<INDArray, Int> {
        val models: MutableList<INDArray> = arrayListOf(myModel.first)
        otherModelPairs.forEach { models.add(it.first) }
        logger.debug { "Found ${models.size} models in total" }
        if (models.size == 1) {
            return myModel
        } else {
            val b = models.map { it.toDoubleMatrix()[0] }
            val newMatrix = Array(1) { DoubleArray(b[0].size) }
            for (i in b[0].indices) {
                val elements = ArrayList<Double>(b.size)
                b.forEach { elements.add(it[i]) }
                newMatrix[0][i] = median(elements)
            }
            return Pair(NDArray(newMatrix), 999999)
        }
    }
}
