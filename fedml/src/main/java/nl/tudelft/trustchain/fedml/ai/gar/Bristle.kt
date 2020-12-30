package nl.tudelft.trustchain.fedml.ai.gar

import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import kotlin.math.max
import kotlin.math.min

/**
 * (practical yet robust) byzantine-resilient decentralized stochastic federated learning
 *
 *
 * byzantine-resilient decentralized stochastic gradient descent federated learning, non i.i.d., history-sensitive (= more robust), practical
 */
class Bristle : AggregationRule() {
    private val TEST_BATCH = 500
    private val NUM_MODELS_EXPLOITATION = 9
    private val NUM_MODELS_EXPLORATION = 1

    @ExperimentalStdlibApi
    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
        logging: Boolean,
    ): INDArray {
        debug(logging) { formatName("BRISTLE") }
        debug(logging) { "Found ${newOtherModels.size} other models" }
        debug(logging) { "oldModel: ${oldModel.getDouble(0)}" }
        val newModel = oldModel.sub(gradient)
        debug(logging) { "newModel: ${newModel.getDouble(0)}" }
        debug(logging) { "otherModels: ${newOtherModels.map { it.value.getDouble(0) }.toCollection(ArrayList())}" }

        val distances = getDistances(oldModel, newModel, newOtherModels, recentOtherModels, logging)
        debug(logging) { "distances: $distances" }
        val exploitationModels = distances
            .keys
            .take(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .toMap()
        debug(logging) { "closeModels: ${exploitationModels.map { it.value.getDouble(0) }.toCollection(ArrayList())}" }

        val explorationModels = distances
            .keys
            .drop(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .shuffled()
            .take(NUM_MODELS_EXPLORATION)
            .toMap()
        debug(logging) { "notCloseModels: ${explorationModels.map { it.value.getDouble(0) }.toCollection(ArrayList())}" }

        val combinedModels = HashMap<Int, INDArray>()
        combinedModels.putAll(exploitationModels)
        combinedModels.putAll(explorationModels)
        debug(logging) { "combinedModels: ${combinedModels.map { it.value.getDouble(0) }.toCollection(ArrayList())}" }
        testDataSetIterator.reset()
        val sample = testDataSetIterator.next(TEST_BATCH)
        val oldLoss = calculateLoss(oldModel, network, sample)
        debug(logging) { "oldLoss: $oldLoss" }
        val oldLossPerClass = calculateLossPerClass(oldModel, network, testDataSetIterator.testBatches)
        debug(logging) { "oldLossPerClass: ${oldLossPerClass.toList()}" }
        val losses = calculateLosses(combinedModels, network, sample)
        debug(logging) { "losses: $losses" }
        val lossesPerClass = calculateLossesPerClass(combinedModels, network, testDataSetIterator.testBatches)
        debug(logging) { "lossesPerClass: ${lossesPerClass.map { Pair(it.key, it.value.toList()) }}" }
        val modelsToWeight = mapLossesToWeight(losses, oldLoss)
        debug(logging) { "modelsToWeight: $modelsToWeight" }
        val modelsPerClassToWeight = mapLossesPerClassToWeight(lossesPerClass, oldLossPerClass, countPerPeer, logging)
        debug(logging) { "modelsPerClassToWeight: $modelsPerClassToWeight" }

        return if (modelsPerClassToWeight.isEmpty()) {
            newModel
        } else {
            weightedAverage(modelsPerClassToWeight, combinedModels, newModel, logging)
        }
    }

    override fun isDirectIntegration(): Boolean {
        return true
    }

    private fun getDistances(
        oldModel: INDArray,
        newModel: INDArray,
        otherModels: Map<Int, INDArray>,
        allOtherModelsBuffer: ArrayDeque<Pair<Int, INDArray>>,
        logging: Boolean,
    ): Map<Int, Double> {
        val distances = hashMapOf<Int, Double>()
        for ((index, otherModel) in otherModels) {
            val min = min(otherModel.distance2(oldModel), otherModel.distance2(newModel))
            debug(logging) { "Distance calculated: $min" }
            distances[index] = min
        }
        for (i in 0 until min(20 - distances.size, allOtherModelsBuffer.size)) {
            val otherModel = allOtherModelsBuffer.elementAt(allOtherModelsBuffer.size - 1 - i)
            distances[1000000 + otherModel.first] =
                min(otherModel.second.distance2(oldModel), otherModel.second.distance2(newModel))
        }
        return distances.toList().sortedBy { (_, value) -> value }.toMap()
    }

    private fun calculateLoss(
        model: INDArray,
        network: MultiLayerNetwork,
        sample: DataSet,
    ): Double {
        network.setParameters(model)
        return network.score(sample)
    }

    private fun calculateLossPerClass(
        model: INDArray,
        network: MultiLayerNetwork,
        testBatches: List<DataSet?>,
    ): Array<Double?> {
        network.setParameters(model)
        return testBatches
            .map { if (it == null) null else network.score(it) }
            .toTypedArray()
    }

    private fun calculateLosses(
        models: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        sample: DataSet,
    ): Map<Int, Double> {
        return models.map { (index, model) ->
            network.setParameters(model)
            Pair(index, network.score(sample))
        }.toMap()
    }

    private fun calculateLossesPerClass(
        models: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        testBatches: List<DataSet?>,
    ): Map<Int, Array<Double?>> {
        return models.map { (index, model) ->
            Pair(
                index, testBatches.map { testBatch ->
                    if (testBatch == null) {
                        null
                    } else {
                        network.setParameters(model)
                        network.score(testBatch)
                    }
                }.toTypedArray()
            )
        }.toMap()
    }

    private fun mapLossesToWeight(otherLosses: Map<Int, Double>, oldLoss: Double): Map<Int, Double> {
        val minWeight = 1
        val maxWeight = 5
        return otherLosses
            .filter { it.value < oldLoss }
            .map { (index, loss) ->
                val weightDistance = maxWeight - minWeight
                val performance = (1 - (loss / oldLoss))
                Pair(index, minWeight + weightDistance * performance)
            }
            .toMap()
    }

    private fun mapLossesPerClassToWeight(
        otherLossesPerClass: Map<Int, Array<Double?>>,
        oldLossPerClass: Array<Double?>,
        countPerPeer: Map<Int, Int>,
        logging: Boolean,
    ): Map<Int, Double> {
        val smallestLossPerPeer = otherLossesPerClass
            .map { (peer, lossPerClass) ->
                Pair(peer, lossPerClass
                    .mapIndexed { label, loss -> Pair(label, loss ?: Double.MAX_VALUE) }
                    .sortedBy { it.second }
                    .take(countPerPeer.getValue(peer))
                    .toMap())
            }.toMap()
        debug(logging) { "smallestLossPerPeer: $smallestLossPerPeer" }

        return smallestLossPerPeer.map { (peer, lossPerLabel) ->
            val smallestOtherLoss = lossPerLabel.map { (_, smallestLosses) -> smallestLosses }.sum()
            val smallestOwnLoss = lossPerLabel.map { (label, _) -> oldLossPerClass[label]!! }.sum()
            Pair(peer, max(0.0, 1.0 + (0 - 0 * (smallestOtherLoss / smallestOwnLoss))))
        }.toMap()
    }

    private fun weightedAverage(
        modelsToWeight: Map<Int, Double>,
        otherModels: Map<Int, INDArray>,
        newModel: INDArray,
        logging: Boolean,
    ): INDArray {
        var arr: INDArray? = null
        modelsToWeight.onEachIndexed { indexAsNum, (indexAsPeer, weight) ->
            if (indexAsNum == 0) {
                arr = otherModels.getValue(indexAsPeer).mul(weight)
            } else {
                arr!!.addi(otherModels.getValue(indexAsPeer).mul(weight))
            }
        }
        arr!!.addi(newModel)
        val totalWeight = modelsToWeight.values.sum() + 1  // + 1 for the new model
        debug(logging) { "totalWeight: $totalWeight" }
        val result = arr!!.div(totalWeight)
        debug(logging) { "weightedAverage: $result" }
        return result
    }
}
