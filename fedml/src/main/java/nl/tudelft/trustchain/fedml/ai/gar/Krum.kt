package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

private val logger = KotlinLogging.logger("Krum")

fun getKrum(models: MutableList<INDArray>, b: Int): Int {
    val distances = Array(models.size) { FloatArray(models.size) }
    for (i in 0 until models.size - 1) {
        distances[i][i] = 9999999.0f
        for (j in i + 1 until models.size) {
            val distance = models[i].distance2(models[j]).toFloat()
            distances[i][j] = distance
            distances[j][i] = distance
        }
    }
    distances[models.size - 1][models.size - 1] = 9999999.0f
    val summedDistances = distances.map {
        val copy = it.copyOf()
        copy.sort()
        copy.take(models.size - b/* - 2*/).sum()
    }
    return summedDistances.indexOf(summedDistances.min())
}

class Krum(private val b: Int) : AggregationRule() {
    override fun integrateParameters(
        myModel: INDArray,
        gradient: INDArray,
        otherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): INDArray {
        logger.debug { formatName("Krum") }
        val models: MutableList<INDArray> = arrayListOf()
        otherModels.forEach { models.add(it) }
        logger.debug { "Found ${models.size} models in total" }
        return if (models.size == 1) {
            myModel.sub(gradient)
        } else {
            val bestCandidate = getKrum(models, b)
            val newModel = myModel.sub(gradient).add(models[bestCandidate]).div(2)
            newModel
        }
    }
}
