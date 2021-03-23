package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.NDArrayStrings2
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.d
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import kotlin.math.*

private const val NUM_MODELS_EXPLOITATION = 20
private const val NUM_MODELS_EXPLORATION = 5
private const val MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING = 20

private typealias Peer = Int
private typealias Class = Int
private typealias SeqAttackPenalty = Double
private typealias Score = Double
private typealias Weight = Double
private typealias Certainty = Double

private val logger = KotlinLogging.logger("Bristle")

/**
 * (practical yet robust) byzantine-resilient decentralized stochastic federated learning
 *
 *
 * byzantine-resilient decentralized stochastic gradient descent federated learning, non i.i.d., history-sensitive (= more robust), practical
 */
class Bristle : AggregationRule() {
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
        val formatter = NDArrayStrings2()
        logger.d(logging) { formatName("BRISTLE") }
        logger.d(logging) { "Found ${newOtherModels.size} other models" }
        logger.d(logging) { "oldModel: ${formatter.format(oldModel)}" }
        val newModel = oldModel.sub(gradient)
        logger.d(logging) { "newModel: ${formatter.format(newModel)}" }
        logger.d(logging) { "otherModels: ${newOtherModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val distances = getDistances(newModel, newOtherModels, recentOtherModels, logging)
        logger.d(logging) { "distances: $distances" }
        val exploitationModels = distances
            .keys
            .take(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .toMap()
        logger.d(logging) { "closeModels: ${exploitationModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val explorationModels = distances
            .keys
            .drop(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .shuffled()
            .take(NUM_MODELS_EXPLORATION)
            .toMap()
        logger.d(logging) { "notCloseModels: ${explorationModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val combinedModels = HashMap<Peer, INDArray>()
        combinedModels.putAll(exploitationModels)
        combinedModels.putAll(explorationModels)
        combinedModels.forEach {
            logger.d(logging) { "combinedModel (${it.key}): ${formatter.format(it.value)}" }
        }
        val peers = combinedModels.keys.sorted().toIntArray()
        testDataSetIterator.reset()

        val myRecalls = calculateRecallPerClass(newModel, network, testDataSetIterator)
        logger.d(logging) { "myRecallPerClass: ${myRecalls.toList()}" }

        val recallsPerPeer = calculateRecallPerClass(combinedModels, network, testDataSetIterator)
        logger.d(logging) { "peerRecallPerClass: ${recallsPerPeer.map { Pair(it.key, it.value.toList()) }}}" }

        val selectedClassesPerPeer = getBestPerformingClassesPerPeer(peers, myRecalls, recallsPerPeer, countPerPeer)
        logger.d(logging) { "selectedClassesPerPeer: ${selectedClassesPerPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightedDiffsPerSelectedPeer = getWeightedDiffsPerSelectedPeer(peers, myRecalls, recallsPerPeer, selectedClassesPerPeer)
        logger.d(logging) { "weightedDiffPerSelectedPeer: ${weightedDiffsPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val seqAttackPenaltiesPerSelectedPeer = getSeqAttackPenaltiesPerSelectedPeer(peers, myRecalls, recallsPerPeer, selectedClassesPerPeer)
        logger.d(logging) { "seqAttackPenaltyPerSelectedPeer: ${seqAttackPenaltiesPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val scoresPerSelectedPeer = getScoresPerSelectedPeer(peers, myRecalls, recallsPerPeer, selectedClassesPerPeer, weightedDiffsPerSelectedPeer, seqAttackPenaltiesPerSelectedPeer, true)
        logger.d(logging) { "scoresPerSelectedPeer: ${scoresPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val certaintiesPerSelectedPeer = getCertaintiesPerSelectedPeer(peers, selectedClassesPerPeer, recallsPerPeer)
        logger.d(logging) { "certaintiesPerSelectedPeer: ${certaintiesPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }



        val certaintyPerSelectedPeer = getCertaintyPerSelectedPeer(peers, recallsPerPeer, selectedClassesPerPeer)
        logger.d(logging) { "certaintyPerSelectedPeer: $certaintyPerSelectedPeer" }



        val weightsPerSelectedPeer = getWeightsPerSelectedPeer(peers, selectedClassesPerPeer, scoresPerSelectedPeer, certaintiesPerSelectedPeer, certaintyPerSelectedPeer)
        logger.d(logging) { "weightsPerSelectedPeer: ${weightsPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightedAverage = weightedAverage(weightsPerSelectedPeer, combinedModels, newModel, logging, testDataSetIterator.labels)
        logger.d(logging) { "weightedAverage: ${formatter.format(weightedAverage)}" }


        val unboundedScoresPerSelectedPeer = getScoresPerSelectedPeer(peers, myRecalls, recallsPerPeer, selectedClassesPerPeer, weightedDiffsPerSelectedPeer, seqAttackPenaltiesPerSelectedPeer, false)
        logger.d(logging) { "unboundedScoresPerSelectedPeer: ${unboundedScoresPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightPerSelectedPeer = getWeightPerSelectedPeer(peers, unboundedScoresPerSelectedPeer, certaintyPerSelectedPeer)
        logger.d(logging) { "weightPerSelectedPeer: $weightPerSelectedPeer" }

        val result = incorporateForeignWeights(peers, weightPerSelectedPeer, combinedModels, weightedAverage, testDataSetIterator.labels, logging)
        logger.d(logging) { "result: ${formatter.format(result)}" }
        return result
    }

    override fun isDirectIntegration(): Boolean {
        return true
    }

    private fun getDistances(
        newModel: INDArray,
        otherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        logging: Boolean,
    ): Map<Int, Double> {
        val distances = hashMapOf<Int, Double>()
        for ((index, otherModel) in otherModels) {
            val min = otherModel.distance2(newModel)
            logger.d(logging) { "Distance calculated: $min" }
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
        val tw = network.outputLayer.paramTable()
        for (index in 0 until tw.getValue("W").columns()) {
            tw.getValue("W").putColumn(index, model.getColumn(index.toLong()).dup())
        }
        network.outputLayer.setParamTable(tw)
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
    ): Map<Peer, DoubleArray> {
        return models.map { (index, model) ->
            Pair(index, calculateRecallPerClass(model, network, testDataSetIterator))
        }.toMap()
    }

    private fun getBestPerformingClassesPerPeer(
        peers: IntArray,
        myRecallPerClass: DoubleArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
        countPerPeer: Map<Int, Int>
    ): Map<Peer, IntArray> {
        return peers.map { peer ->
            val selectionSize = countPerPeer.getOrDefault(peer, myRecallPerClass.size)
            val selectedClasses = if (selectionSize == peerRecallPerClass.getValue(peer).size)
                (0 until selectionSize).toList().toIntArray()
            else
                peerRecallPerClass.getValue(peer)
                    .mapIndexed { label, recall -> Pair(label, recall) }
                    .sortedByDescending { it.second }
                    .take(selectionSize)  // Attackers have a negative index => assume that they have the same classes as the current peer to maximize attack strength
                    .map { it.first }
                    .toIntArray()
            Pair(peer, selectedClasses)
        }.toMap()
    }

    private fun getWeightedDiffsPerSelectedPeer(
        peers: IntArray,
        myRecallPerClass: DoubleArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
        selectedClassesPerPeer: Map<Peer, IntArray>
    ): Map<Peer, Map<Class, Double>> {
        return peers.map { peer ->
            val weightDiffsPerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                Pair(selectedClass, abs(myRecallPerClass[selectedClass] - peerRecallPerClass.getValue(peer)[selectedClass]) * 10)
            }.toMap()
            Pair(peer, weightDiffsPerSelectedClass)
        }.toMap()
    }

    private fun getSeqAttackPenaltiesPerSelectedPeer(
        peers: IntArray,
        myRecallPerClass: DoubleArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
        selectedClassesPerPeer: Map<Peer, IntArray>
    ): Map<Peer, Map<Class, SeqAttackPenalty>> {
        val previousPeerRecalls = HashSet<DoubleArray>()
        return peers.map { peer ->
            val seqAttackPenaltiesPerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                var seqAttackPenalty = 0.0
                for (recalls in previousPeerRecalls) {
                    seqAttackPenalty += max(0.0, 2 * (myRecallPerClass[selectedClass] - recalls[selectedClass]))
                }
                Pair(selectedClass, seqAttackPenalty)
            }.toMap()
            previousPeerRecalls.add(peerRecallPerClass.getValue(peer))
            Pair(peer, seqAttackPenaltiesPerSelectedClass)
        }.toMap()
    }

    private fun getScoresPerSelectedPeer(
        peers: IntArray,
        myRecallPerClass: DoubleArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
        selectedClassesPerPeer: Map<Peer, IntArray>,
        weightedDiffPerSelectedPeer: Map<Peer, Map<Class, Double>>,
        seqAttackPenaltyPerSelectedPeer: Map<Peer, Map<Class, SeqAttackPenalty>>,
        bounded: Boolean
    ): Map<Peer, Map<Class, Score>> {
        return peers.map { peer ->
            val scorePerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                val myRecall = myRecallPerClass[selectedClass]
                val peerRecall = peerRecallPerClass.getValue(peer)[selectedClass]
                val weightedDiff = weightedDiffPerSelectedPeer.getValue(peer).getValue(selectedClass)
                val seqAttackPenalty = seqAttackPenaltyPerSelectedPeer.getValue(peer).getValue(selectedClass)
                val score = if (peerRecall > myRecall) weightedDiff.pow(3 + myRecall) else -1.0 * weightedDiff.pow(4 + myRecall) * (1 + seqAttackPenalty)
                Pair(selectedClass, if (bounded) max(-100.0, score) else score)
            }.toMap()
            Pair(peer, scorePerSelectedClass)
        }.toMap()
    }

    private fun getCertaintiesPerSelectedPeer(
        peers: IntArray,
        selectedClassesPerPeer: Map<Peer, IntArray>,
        recallsPerPeer: Map<Peer, DoubleArray>
    ): Map<Peer, Map<Class, Certainty>> {
        return peers.map { peer ->
            val weightPerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                val certainty = clamp((recallsPerPeer.getValue(peer)[selectedClass] - 0.2) * 4)
                Pair(selectedClass, certainty)
            }.toMap()
            Pair(peer, weightPerSelectedClass)
        }.toMap()
    }

    private fun getWeightsPerSelectedPeer(
        peers: IntArray,
        selectedClassesPerPeer: Map<Peer, IntArray>,
        scoresPerSelectedPeer: Map<Peer, Map<Class, Score>>,
        certaintiesPerSelectedPeer: Map<Peer, Map<Class, Certainty>>,
        certaintyPerSelectedPeer: Map<Peer, Double>
    ): Map<Peer, Map<Class, Weight>> {
        return peers.map { peer ->
            val weightPerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                var sigmoid = 1 / (1 + exp(-scoresPerSelectedPeer.getValue(peer).getValue(selectedClass) / 100))  // sigmoid function
                sigmoid *= 10  // sigmoid from 0 -> 1 to 0 -> 10
                sigmoid -= 4  // sigmoid from 0 -> 10 to -4 -> 6
                sigmoid = max(0.0, sigmoid)  // no negative weights
                sigmoid *= certaintyPerSelectedPeer.getValue(peer)/*certaintiesPerSelectedPeer.getValue(peer).getValue(selectedClass)*/
                Pair(selectedClass, sigmoid)
            }.toMap()
            Pair(peer, weightPerSelectedClass)
        }.toMap()
    }

    private fun weightedAverage(
        weightsPerSelectedPeer: Map<Peer, Map<Class, Weight>>,
        combinedModels: Map<Int, INDArray>,
        newModel: INDArray,
        logging: Boolean,
        labels: List<String>,
    ): INDArray {
        val arr = newModel.dup()
        val totalWeights = DoubleArray(newModel.columns()) { 1.0 }
        weightsPerSelectedPeer.forEach { (peer, weights) ->
            weights.forEach { (selectedClass, weight) ->
                val correspondingClass = labels[selectedClass].toInt()
                val currentClassParameters = arr.getColumn(correspondingClass.toLong())
                val extraClassParameters = combinedModels[peer]!!.getColumn(correspondingClass.toLong()).mul(weight)
                val newClassParameters = currentClassParameters.add(extraClassParameters)
                arr.putColumn(correspondingClass, newClassParameters)
                totalWeights[correspondingClass] += weight
            }
        }
        logger.d(logging) { "totalWeights: ${totalWeights.contentToString()}" }
        totalWeights.forEachIndexed { index, weight ->
            if (weight != 1.0) {  // optimization
                val scaledColumn = arr.getColumn(index.toLong()).div(weight)
                arr.putColumn(index, scaledColumn/*.subi(newModel.meanNumber())*/)
            }
        }
        return arr
    }

    private fun getCertaintyPerSelectedPeer(
        peers: IntArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
        selectedClassesPerPeer: Map<Peer, IntArray>,
    ): Map<Peer, Double> {
        return peers.map { peer ->
            val selectedClasses = selectedClassesPerPeer.getValue(peer)
            val recallPerSelectedClasses = peerRecallPerClass.getValue(peer).sliceArray(selectedClasses.toList())
            val average = recallPerSelectedClasses.average()
            val std = recallPerSelectedClasses.std()
            Pair(peer, max(0.0, average - std))
        }.toMap()
    }

    private fun getWeightPerSelectedPeer(
        peers: IntArray,
        unboundedScoresPerSelectedPeer: Map<Peer, Map<Class, Score>>,
        certaintyPerSelectedPeer: Map<Peer, Double>
    ): Map<Peer, Double> {
        return peers.map { peer ->
            val totalScore = unboundedScoresPerSelectedPeer.getValue(peer).values.sum()
            var sigmoid = 1 / (1 + exp(-totalScore / 100))  // sigmoid function
            sigmoid *= 10  // sigmoid from 0 -> 1 to 0 -> 10
            sigmoid -= 4  // sigmoid from 0 -> 10 to -4 -> 6
            sigmoid = max(0.0, sigmoid)  // no negative weights
            sigmoid *= certaintyPerSelectedPeer.getValue(peer)
            Pair(peer, sigmoid)
        }.toMap()
    }

    private fun incorporateForeignWeights(
        peers: IntArray,
        weightPerSelectedPeer: Map<Peer, Double>,
        combinedModels: Map<Peer, INDArray>,
        weightedAverage: INDArray,
        labels: List<String>,
        logging: Boolean
    ): INDArray {
        val arr = weightedAverage.dup()
        val foreignLabels = (0 until weightedAverage.columns()).subtract(labels.map { it.toInt() })
        logger.d(logging) { "foreignLabels: $foreignLabels" }
        peers.forEach { peer ->
            foreignLabels.forEach { label ->
                val targetColumn = combinedModels.getValue(peer).getColumn(label.toLong())
                arr.putColumn(label, arr.getColumn(label.toLong()).add(targetColumn.mul(weightPerSelectedPeer.getValue(peer))))
            }
        }
        val totalWeight = weightPerSelectedPeer.values.sum() + 1   // + 1 for the current node
        logger.d(logging) { "totalWeight: $totalWeight" }
        foreignLabels.forEach { label ->
            arr.putColumn(label, arr.getColumn(label.toLong()).div(totalWeight))
        }
        return arr
    }

    private fun clamp(num: Double, min: Double = 0.0, max: Double = 1.0): Double {
        return max(min, min(max, num))
    }

    private fun DoubleArray.std(): Double {
        val avg = this.average()
        val std = this.fold(0.0) { a, b -> a + (b - avg).pow(2) }
        return sqrt(std / this.size)
    }
}
