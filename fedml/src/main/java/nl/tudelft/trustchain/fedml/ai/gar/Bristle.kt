package nl.tudelft.trustchain.fedml.ai.gar

import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import kotlin.math.*

private const val NUM_MODELS_EXPLOITATION = 50
private const val NUM_MODELS_EXPLORATION = 5
private const val MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING = 20

/**
 * (practical yet robust) byzantine-resilient decentralized stochastic federated learning
 *
 *
 * byzantine-resilient decentralized stochastic gradient descent federated learning, non i.i.d., history-sensitive (= more robust), practical
 */
class Bristle : AggregationRule() {

    private fun average(models: HashMap<Int, INDArray>) : INDArray {
        var arr: INDArray? = null
        models.onEachIndexed { indexAsNum, (_, model) ->
            if (indexAsNum == 0) {
                arr = model.dup()
            } else {
                arr!!.addi(model)
            }
        }
        return arr!!.divi(models.size)
    }
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
        val models = HashMap<Int, INDArray>()
        models[-1] = oldModel.sub(gradient)
        models.putAll(newOtherModels)
        debug(logging) { "Found ${models.size} models in total" }
        return average(models)
        /*debug(logging) { "Found ${newOtherModels.size} other models" }
        debug(logging) { "oldModel: ${oldModel.getFloat(0)}" }
        val newModel = oldModel.sub(gradient)
        debug(logging) { "newModel: ${newModel.getFloat(0)}" }
        debug(logging) { "otherModels: ${newOtherModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val distances = getDistances(oldModel, newModel, newOtherModels, recentOtherModels, logging)
        debug(logging) { "distances: $distances" }
        val exploitationModels = distances
            .keys
            .take(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .toMap()
        debug(logging) { "closeModels: ${exploitationModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val explorationModels = distances
            .keys
            .drop(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .shuffled()
            .take(NUM_MODELS_EXPLORATION)
            .toMap()
        debug(logging) { "notCloseModels: ${explorationModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val combinedModels = HashMap<Int, INDArray>()
        combinedModels.putAll(exploitationModels)
        combinedModels.putAll(explorationModels)
        debug(logging) { "combinedModels: ${combinedModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }
        testDataSetIterator.reset()
        val myRecallPerClass = calculateRecallPerClass(newModel, network, testDataSetIterator)
        debug(logging) { "myRecallPerClass: ${myRecallPerClass.toList()}" }
        val peerRecallPerClass = calculateRecallPerClass(combinedModels, network, testDataSetIterator)
        debug(logging) { "peerRecallPerClass: ${peerRecallPerClass.map { Pair(it.key, it.value.toList()) }}" }
        val modelsPerClassToWeight = mapRecallPerClassToWeight(myRecallPerClass, peerRecallPerClass, countPerPeer, logging)
        debug(logging) { "modelsPerClassToWeight: $modelsPerClassToWeight" }

        return if (modelsPerClassToWeight.isEmpty()) {
            newModel
        } else {
            weightedAverage(modelsPerClassToWeight, combinedModels, newModel, logging)
        }*/
    }

    override fun isDirectIntegration(): Boolean {
        return true
    }

    private fun getDistances(
        oldModel: INDArray,
        newModel: INDArray,
        otherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        logging: Boolean,
    ): Map<Int, Double> {
        val distances = hashMapOf<Int, Double>()
        for ((index, otherModel) in otherModels) {
            val min = otherModel.distance2(newModel)
            debug(logging) { "Distance calculated: $min" }
            distances[index] = min
        }
        for (i in 0 until min(MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING - distances.size, recentOtherModels.size)) {
            val otherModel = recentOtherModels.elementAt(recentOtherModels.size - 1 - i)
            distances[1100000 + otherModel.first] = otherModel.second.distance2(newModel)
        }
        return distances.toList().sortedBy { (_, value) -> value }.toMap()
    }

    private fun calculateRecallPerClass(
        model: INDArray,
        network: MultiLayerNetwork,
        testDataSetIterator: CustomDataSetIterator,
    ): DoubleArray {
        network.outputLayer.setParam("W", model)
        val evaluations = arrayOf(Evaluation())
        network.doEvaluation(testDataSetIterator, *evaluations)
        return testDataSetIterator.labels.map {
            evaluations[0].recall(it.toInt())
        }.toDoubleArray()
    }

    private fun calculateRecallPerClass(
        models: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: CustomDataSetIterator,
    ): Map<Int, DoubleArray> {
        return models.map { (index, model) ->
            Pair(index, calculateRecallPerClass(model, network, testDataSetIterator))
        }.toMap()
    }

    private fun mapRecallPerClassToWeight(
        myRecallPerClass: DoubleArray,
        peerRecallPerClass: Map<Int, DoubleArray>,
        countPerPeer: Map<Int, Int>,
        logging: Boolean,
    ): Map<Int, Double> {
        val seqAttackPenalty = myRecallPerClass.map { 0.0 }.toMutableList()
        val myRecallAverage = myRecallPerClass.average()

        val selectedClassesAndRecallPerPeer = peerRecallPerClass.map { (peer, recallPerClass) ->
            val selectionSize = countPerPeer.getOrDefault(peer, myRecallPerClass.size)

            val selectedClassesAndRecall = if (selectionSize == recallPerClass.size)
                recallPerClass
                    .mapIndexed { label, recall -> Pair(label, recall) }
                    .toMap()
            else
                recallPerClass
                    .mapIndexed { label, recall -> Pair(label, recall) }
                    .sortedByDescending { it.second }
                    .take(selectionSize)  // Attackers have a negative index => assume that they have the same classes as the current peer to maximize attack strength
                    .toMap()

            debug(logging) { "selectedClassesAndRecall ($peer): $selectedClassesAndRecall" }

            Pair(peer, selectedClassesAndRecall)
        }.toMap()

        val weightsPerPeer = selectedClassesAndRecallPerPeer.map { (peer, recallPerClass) ->
            val weightPerClass = recallPerClass.map { (clazz, peerRecall) ->
                val myRecall = myRecallPerClass[clazz]
                val untrainedModelAmbiguity = if (myRecall >= peerRecall) myRecallAverage else 1.0
                val weightedDiff = abs(myRecall - peerRecall) * 10 * untrainedModelAmbiguity
                val w = if (peerRecall >= myRecall) {
                    weightedDiff.pow(2 + myRecall * 1.5)
                } else {
                    -weightedDiff.pow(3 + myRecall * 1.5) * (1 + max(0.0, seqAttackPenalty[clazz]))
                }
                debug(logging) { "peer: $peer, class: $clazz, myRecall: $myRecall, peerRecall: $peerRecall, weightedDiff: $weightedDiff, untrainedModelAmbiguity: $untrainedModelAmbiguity, w: $w, seqAttackPenalty: ${seqAttackPenalty[clazz]}" }
                seqAttackPenalty[clazz] += 2 * (myRecall - peerRecall)
                w
            }
            Pair(peer, weightPerClass)
        }

        return weightsPerPeer.map { (peer, weightPerClass) ->
            debug(logging) { "weightPerClass: ${weightPerClass.toList()}"}
            val weightSum = weightPerClass.sum()
            debug(logging) { "weightSum: $weightSum"}
            if (weightSum < -1000) {
                Pair(peer, 0.0)
            } else {
                val selectedRecalls = selectedClassesAndRecallPerPeer.getValue(peer).values
                val avg = selectedRecalls.average()
                debug(logging) { "avg: $avg"}
                val std = selectedRecalls.std()
                debug(logging) { "std: $std"}
                val certainty = max(0.0, avg - std * 2)
                debug(logging) { "certainty: $certainty"}

                var sigmoid = 1 / (1 + exp(-weightSum / 100))  // sigmoid function
                debug(logging) { "sigmoid: $sigmoid"}
                sigmoid *= 10  // sigmoid from 0 -> 1 to 0 -> 10
                debug(logging) { "sigmoid: $sigmoid"}
                sigmoid -= 4  // sigmoid from 0 -> 10 to -4 -> 6
                debug(logging) { "sigmoid: $sigmoid"}
                sigmoid = max(0.0, sigmoid)  // no negative weights
                sigmoid *= certainty
                debug(logging) { "sigmoid: $sigmoid"}
                Pair(peer, sigmoid)
            }
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
        val result = arr!!.divi(totalWeight)
        debug(logging) { "weightedAverage: $result" }
        return result
    }

    companion object {
        const val PRE_TRAIN = 25
    }
}

private fun <E> Collection<E>.std(): Double {
    val avg = (this as Collection<Double>).average()
    val std = (this as Collection<Double>).fold(0.0) { a, b -> a + (b - avg).pow(2) }
    return sqrt(std / this.size)
}
