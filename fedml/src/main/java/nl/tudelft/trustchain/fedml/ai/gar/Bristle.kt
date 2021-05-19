package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.NDArrayStrings2
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.d
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import kotlin.math.*

private const val MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING = 10
private const val ALPHA = 0.4

private typealias Peer = Int
private typealias Class = Int
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

        val startTime = System.currentTimeMillis()
        val distances = getDistances(newModel, newOtherModels, recentOtherModels, true)
        logger.d(logging) { "distances: $distances" }

        val split = listOf(distances.take(distances.size / 3), distances.subList(distances.size / 3, (distances.size * (2.0/3.0)).toInt()), distances.takeLast(distances.size / 3))

        val fl = round((1.0 - ALPHA).pow(2) * MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING).toInt()
        logger.d(logging) { "fl: $fl" }
        val fm = round((-2 * ALPHA.pow(2) + 2 * ALPHA) * MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING).toInt()
        logger.d(logging) { "fm: $fm" }
        val fh = round(ALPHA.pow(2) * MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING).toInt()
        logger.d(logging) { "fh: $fh" }

        val combinedModels = HashMap<Peer, INDArray>()
        combinedModels.putAll(distances.shuffled().take(10).map { Pair(it.first, newOtherModels.getValue(it.first)) }.toMap())
        /*combinedModels.putAll(
            split[0].shuffled().take(fl).map { Pair(it.first, newOtherModels.getValue(it.first)) }.toMap(),
        )
        combinedModels.putAll(
            split[1].shuffled().take(fm).map { Pair(it.first, newOtherModels.getValue(it.first)) }.toMap(),
        )
        combinedModels.putAll(
            split[2].shuffled().take(fh).map { Pair(it.first, newOtherModels.getValue(it.first)) }.toMap()
        )*/
        val peers = combinedModels.keys.sorted().toIntArray()
        val endTime = System.currentTimeMillis()
        logger.d(logging) { "Timing 0: ${endTime - startTime}" }
        testDataSetIterator.reset()
        val startTime2 = System.currentTimeMillis()

        val familiarClasses = testDataSetIterator.labels.map { it.toInt() }
        logger.d(logging) { "familiarClasses: $familiarClasses" }
        val foreignLabels = (0 until newModel.columns()).subtract(familiarClasses).toList()
        logger.d(logging) { "foreignLabels: $foreignLabels" }

        val myF1s = calculateF1PerClass(newModel, network, testDataSetIterator, familiarClasses)
        logger.d(logging) { "myF1s: ${myF1s.toList()}" }

        val f1sPerPeer = calculateF1PerClass(combinedModels, network, testDataSetIterator, familiarClasses)
        logger.d(logging) { "f1sPerPeer: ${f1sPerPeer.map { Pair(it.key, it.value.toList()) }}}" }

        val selectedClassesPerPeer = getBestPerformingClassesPerPeer(peers, myF1s, f1sPerPeer, countPerPeer)
        logger.d(logging) { "selectedClassesPerPeer: ${selectedClassesPerPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightedDiffsPerSelectedPeer = getWeightedDiffsPerSelectedPeer(peers, myF1s, f1sPerPeer, selectedClassesPerPeer)
        logger.d(logging) { "weightedDiffPerSelectedPeer: ${weightedDiffsPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val scoresPerPeer = getScoresPerPeer(peers, myF1s, f1sPerPeer, weightedDiffsPerSelectedPeer, true)
        logger.d(logging) { "scoresPerSelectedPeer: ${scoresPerPeer.map { Pair(it.key, it.value.toList()) }}" }



        val certaintyPerSelectedPeer = getCertaintyPerSelectedPeer(peers, f1sPerPeer, selectedClassesPerPeer)
        logger.d(logging) { "certaintyPerSelectedPeer: $certaintyPerSelectedPeer" }



        val weightsPerSelectedPeer = getWeightsPerSelectedPeer(peers, scoresPerPeer, certaintyPerSelectedPeer)
        logger.d(logging) { "weightsPerSelectedPeer: ${weightsPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightedAverage = weightedAverage(weightsPerSelectedPeer, combinedModels, newModel, true, familiarClasses)
        logger.d(logging) { "weightedAverage: ${formatter.format(weightedAverage)}" }


        val unboundedScoresPerSelectedPeer = getScoresPerPeer(peers, myF1s, f1sPerPeer, weightedDiffsPerSelectedPeer, false)
        logger.d(logging) { "unboundedScoresPerSelectedPeer: ${unboundedScoresPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightPerSelectedPeer = getWeightPerSelectedPeer(peers, unboundedScoresPerSelectedPeer, certaintyPerSelectedPeer)
        logger.d(logging) { "weightPerSelectedPeer: $weightPerSelectedPeer" }

        val result = incorporateForeignWeights(peers, weightPerSelectedPeer, combinedModels, weightedAverage, foreignLabels, true)
        logger.d(logging) { "result: ${formatter.format(result)}" }
        val endTime2 = System.currentTimeMillis()
        logger.d(logging) { "Timing 1: ${endTime2 - startTime2}" }
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
    ): List<Pair<Int, Double>> {
        val distances = hashMapOf<Int, Double>()
        for ((index, otherModel) in otherModels) {
            val min = otherModel.distance2(newModel)
            logger.d(logging) { "Distance calculated: $min" }
            distances[index] = min
        }
//        for (i in 0 until min(MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING - distances.size, recentOtherModels.size)) {
//            val otherModel = recentOtherModels.elementAt(recentOtherModels.size - 1 - i)
//            distances[1100000 + otherModel.first] = otherModel.second.distance2(newModel)
//        }
        return distances.toList().sortedBy { (_, value) -> value }
    }

    private fun calculateF1PerClass(
        model: INDArray,
        network: MultiLayerNetwork,
        testDataSetIterator: CustomDataSetIterator,
        familiarClasses: List<Class>,
    ): Map<Class, Double> {
        val tw = network.outputLayer.paramTable()
        for (index in 0 until tw.getValue("W").columns()) {
            tw.getValue("W").putColumn(index, model.getColumn(index.toLong()).dup())
        }
        network.outputLayer.setParamTable(tw)
        val evaluations = arrayOf(Evaluation())
        network.doEvaluation(testDataSetIterator, *evaluations)
        return familiarClasses.map { clz ->
            Pair(clz, evaluations[0].f1(clz))
        }.toMap()
    }

    private fun calculateF1PerClass(
        models: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: CustomDataSetIterator,
        familiarClasses: List<Class>,
    ): Map<Peer, Map<Class, Double>> {
        return models.map { (index, model) ->
            Pair(index, calculateF1PerClass(model, network, testDataSetIterator, familiarClasses))
        }.toMap()
    }

    private fun getBestPerformingClassesPerPeer(
        peers: IntArray,
        myF1PerClass: Map<Class, Double>,
        peerF1PerClass: Map<Peer, Map<Class, Double>>,
        countPerPeer: Map<Int, Int>
    ): Map<Peer, IntArray> {
        return peers.map { peer ->
            val selectionSize = countPerPeer.getOrDefault(peer, myF1PerClass.size)
            val selectedClasses = if (selectionSize == peerF1PerClass.getValue(peer).size)
                (0 until selectionSize).toList().toIntArray()
            else
                peerF1PerClass.getValue(peer)
                    .toList()
                    .sortedByDescending { it.second }
                    .take(selectionSize)  // Attackers have a negative index => assume that they have the same classes as the current peer to maximize attack strength
                    .map { it.first }
                    .toIntArray()
            Pair(peer, selectedClasses)
        }.toMap()
    }

    private fun getWeightedDiffsPerSelectedPeer(
        peers: IntArray,
        myF1PerClass: Map<Class, Double>,
        peerF1PerClass: Map<Peer, Map<Class, Double>>,
        selectedClassesPerPeer: Map<Peer, IntArray>,
    ): Map<Peer, Map<Class, Double>> {
        return peers.map { peer ->
            val weightDiffsPerFamiliarClass = selectedClassesPerPeer.getValue(peer).map { selectedFamiliarClass ->
                Pair(selectedFamiliarClass, abs(myF1PerClass.getValue(selectedFamiliarClass) - peerF1PerClass.getValue(peer).getValue(selectedFamiliarClass)) * 10)
            }.toMap()
            Pair(peer, weightDiffsPerFamiliarClass)
        }.toMap()
    }

    private fun getScoresPerPeer(
        peers: IntArray,
        myF1PerClass: Map<Class, Double>,
        peerF1PerClass: Map<Peer, Map<Class, Double>>,
        weightedDiffPerSelectedPeer: Map<Peer, Map<Class, Double>>,
        bounded: Boolean
    ): Map<Peer, Map<Class, Score>> {
        return peers.map { peer ->
            val scorePerFamiliarClass = weightedDiffPerSelectedPeer.getValue(peer).map { (selectedFamiliarClass, weightedDiff) ->
                val myF1 = myF1PerClass.getValue(selectedFamiliarClass)
                val peerF1 = peerF1PerClass.getValue(peer).getValue(selectedFamiliarClass)
                val score = if (peerF1 > myF1) weightedDiff.pow(3 + myF1) else -1.0 * weightedDiff.pow(4 + myF1)
                Pair(selectedFamiliarClass, if (bounded) max(-100.0, score) else score)
            }.toMap()
            Pair(peer, scorePerFamiliarClass)
        }.toMap()
    }

    private fun getWeightsPerSelectedPeer(
        peers: IntArray,
        scoresPerPeer: Map<Peer, Map<Class, Score>>,
        certaintyPerSelectedPeer: Map<Peer, Double>
    ): Map<Peer, Map<Class, Weight>> {
        return peers.map { peer ->
            val weightPerFamiliarClass = scoresPerPeer.getValue(peer).map { (familiarClass, score) ->
                var sigmoid = 1 / (1 + exp(-score / 100))  // sigmoid function
                sigmoid *= 10  // sigmoid from 0 -> 1 to 0 -> 10
                sigmoid -= 4  // sigmoid from 0 -> 10 to -4 -> 6
                sigmoid = max(0.0, sigmoid)  // no negative weights
                sigmoid *= certaintyPerSelectedPeer.getValue(peer)
                Pair(familiarClass, sigmoid)
            }.toMap()
            Pair(peer, weightPerFamiliarClass)
        }.toMap()
    }

    private fun weightedAverage(
        weightsPerSelectedPeer: Map<Peer, Map<Class, Weight>>,
        combinedModels: Map<Int, INDArray>,
        newModel: INDArray,
        logging: Boolean,
        labels: List<Int>,
    ): INDArray {
        val arr = newModel.dup()
        val totalWeights = DoubleArray(newModel.columns()) { 1.0 }
        weightsPerSelectedPeer.forEach { (peer, weights) ->
            weights.forEach { (familiarClass, weight) ->
                val currentClassParameters = arr.getColumn(familiarClass.toLong())
                val extraClassParameters = combinedModels[peer]!!.getColumn(familiarClass.toLong()).mul(weight)
                val newClassParameters = currentClassParameters.add(extraClassParameters)
                arr.putColumn(familiarClass, newClassParameters)
                totalWeights[familiarClass] += weight
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
        peerF1sPerClass: Map<Peer, Map<Class, Double>>,
        selectedClassesPerPeer: Map<Peer, IntArray>,
    ): Map<Peer, Double> {
        return peers.map { peer ->
            val selectedClasses = selectedClassesPerPeer.getValue(peer)
            val f1PerSelectedClasses = selectedClasses.map { clz -> peerF1sPerClass.getValue(peer).getValue(clz) }.toDoubleArray()
            val average = f1PerSelectedClasses.average()
            val std = f1PerSelectedClasses.std()
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
        foreignLabels: List<Int>,
        logging: Boolean
    ): INDArray {
        val arr = weightedAverage.dup()
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
