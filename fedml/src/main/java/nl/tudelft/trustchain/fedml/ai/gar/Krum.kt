package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.concurrent.ConcurrentLinkedDeque

private val logger = KotlinLogging.logger("Krum")

fun getKrum(models: List<INDArray>, b: Int): Int {
    val distances = Array(models.size) { FloatArray(models.size) }
    for (i in 0 until models.size) {
        distances[i][i] = 9999999.0f
        for (j in i + 1 until models.size) {
            val distance = models[i].distance2(models[j]).toFloat()
            distances[i][j] = distance
            distances[j][i] = distance
        }
    }
    val summedDistances = distances.map {
        val sorted = it.sorted()
        sorted.take(models.size - b - 2 - 1).sum()  // The additional -1 is because a peer is not a neighbor of itself
    }
    return summedDistances.indexOf(summedDistances.minOrNull()!!)
}

class Krum(private val b: Int) : AggregationRule() {
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
        debug(logging) { formatName("Krum") }
        val models = HashMap<Int, INDArray>()
        models.putAll(newOtherModels)
        debug(logging) { "Found ${models.size} models in total" }
        return if (models.size + 1 <= b + 2 + 1) {  // The additional +1 is because we need to add the current peer itself
            debug(logging) { "Not using KRUM rule because not enough models found..." }
            oldModel.sub(gradient)
        } else {
            val bestCandidate = getKrum(models.map { it.value }.toList(), b)
            val newModel = oldModel.sub(gradient).add(models[bestCandidate]).div(2)
            newModel
        }
    }

    override fun isDirectIntegration(): Boolean {
        return false
    }
}
