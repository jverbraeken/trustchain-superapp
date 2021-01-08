package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaClientToServer
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaServerToClient
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread
import kotlin.math.round
import kotlin.random.Random

private val logger = KotlinLogging.logger("SimulatedRunner")

private const val SIZE_RECENT_OTHER_MODELS = 20
private const val ONLY_EVALUATE_FIRST_NODE = true

class SimulatedRunner : Runner() {
    override fun run(baseDirectory: File, _unused: Int, _unused2: MLConfiguration) {
        simulate(baseDirectory, 0)
    }

    fun simulate(
        baseDirectory: File,
        automationPart: Int,
    ) {
        val job = SupervisorJob()
        val scope = CoroutineScope(Dispatchers.Default + job)
        scope.launch {
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "simulated",
                listOf(
                    "before or after averaging",
                    "#peers included in current batch"
                )
            )
            try {
                val automation = loadAutomation(baseDirectory)
                logger.debug { "Automation loaded" }
                val (configs, figureNames) = generateConfigs(automation, automationPart)
                logger.debug { "Configs generated" }

                for (figure in configs.indices) {
                    val figureName = figureNames[figure]
                    val figureConfig = configs[figure]

                    for (test in figureConfig.indices) {
                        val testConfig = figureConfig[test]
                        logger.error { "Going to test: $figureName - ${testConfig[0].trainConfiguration.gar.id}" }
                        evaluationProcessor.newSimulation("$figureName - ${testConfig[0].trainConfiguration.gar.id}", testConfig)

                        // All these things have to be initialized before any of the runner threads start
                        val toServerMessageBuffers = testConfig.map { CopyOnWriteArrayList<MsgPsiCaClientToServer>() }.toTypedArray()
                        val toClientMessageBuffers = testConfig.map { CopyOnWriteArrayList<MsgPsiCaServerToClient>() }.toTypedArray()
                        val sraKeyPairs = testConfig.mapIndexed { i, _ -> SRAKeyPair.create(bigPrime, java.util.Random(i.toLong())) }.toTypedArray()
                        val randoms = testConfig.mapIndexed { i, _ -> Random(i) }.toTypedArray()
                        val newOtherModelBuffers = testConfig.map { ConcurrentHashMap<Int, INDArray>() }.toTypedArray()
                        val newOtherModelBuffersTemp = testConfig.map { ConcurrentHashMap<Int, INDArray>() }.toTypedArray()
                        val recentOtherModelsBuffers = testConfig.map { ArrayDeque<Pair<Int, INDArray>>() }.toTypedArray()
                        val datasets = testConfig.map { it.dataset }.toTypedArray()
                        val datasetIteratorConfigurations = testConfig.map { it.datasetIteratorConfiguration }.toTypedArray()
                        val behaviors = testConfig.map { it.trainConfiguration.behavior }.toTypedArray()
                        val iterationsBeforeEvaluations =
                            testConfig.map { it.trainConfiguration.iterationsBeforeEvaluation!! }.toTypedArray()
                        val iterationsBeforeSendings =
                            testConfig.map { it.trainConfiguration.iterationsBeforeSending!! }.toTypedArray()
                        val networks = testConfig.zip(datasets).mapIndexed { i, (nodeConfig, nodeDataset) ->
                            generateNetwork(
                                nodeDataset,
                                nodeConfig.nnConfiguration,
                                i
                            )
                        }.toTypedArray()
                        val joiningLateRemainingIterations =
                            testConfig.zip(iterationsBeforeSendings)
                                .map { it.first.trainConfiguration.joiningLate.rounds * it.second }
                                .toTypedArray()
                        val slowdownRemainingIterations = testConfig.map { 0 }.toTypedArray()
                        val oldParams: Array<INDArray> = networks.map { it.params().dup() }.toTypedArray()
                        val newParams: Array<INDArray> = testConfig.map { NDArray(networks[0].params().shape().map { it.toInt() }.toIntArray()) }.toTypedArray()
                        val gradients: Array<INDArray> = testConfig.map { NDArray(networks[0].params().shape().map { it.toInt() }.toIntArray()) } .toTypedArray()
                        val iters = testConfig.mapIndexed { i, _ ->
                            getDataSetIterators(
                                datasets[i],
                                datasetIteratorConfigurations[i],
                                i.toLong() * 10,
                                baseDirectory,
                                behaviors[i]
                            )
                        }.toTypedArray()
                        val iterTrains = iters.map { it[0] }.toTypedArray()
                        val iterTrainFulls = iters.map { it[1] }.toTypedArray()
                        val iterTests = iters.map { it[2] }.toTypedArray()
                        val iterTestFulls = iters.map { it[3] }.toTypedArray()

                        val countPerPeers = ConcurrentHashMap<Int, Map<Int, Int>>()
                        val threads = testConfig.mapIndexed { i, _ ->
                            thread {
                                countPerPeers[i] = getSimilarPeers(
                                    iterTrains[i],
                                    sraKeyPairs[i],
                                    toServerMessageBuffers,
                                    toClientMessageBuffers,
                                    i
                                )
                            }
                        }
                        threads.forEach { it.join() }
                        networks[0].setListeners(ScoreIterationListener(printScoreIterations))

                        var epochEnd = true
                        var epoch = -1
                        val start = System.currentTimeMillis()




                        for (iteration in 0 until testConfig[0].trainConfiguration.maxIteration.value) {
                            if (epochEnd) {
                                epoch++
                                logger.debug { "Epoch: $epoch" }
                                epochEnd = false
                                iterTrains.forEach { it.reset() }
                                iterTrainFulls.forEach { it.reset() }
                            }
                            networks.forEachIndexed { i, n ->
                                newParams[i] = n.params().dup()
                                gradients[i] = oldParams[i].sub(newParams[i])
                            }
                            newOtherModelBuffers.forEachIndexed { i, map -> map.putAll(newOtherModelBuffersTemp[i]) }
                            newOtherModelBuffersTemp.forEach { it.clear() }
                            logger.debug { "Iteration: $iteration" }

                            for (nodeIndex in testConfig.indices) {
                                logger.debug { "Node: $nodeIndex" }
                                val nodeConfig = testConfig[nodeIndex]
                                val network = networks[nodeIndex]
                                val countPerPeer = countPerPeers[nodeIndex]!!
                                val oldParam = oldParams[nodeIndex]
                                val gradient = gradients[nodeIndex]
                                val random = randoms[nodeIndex]
                                val newOtherModelBuffer = newOtherModelBuffers[nodeIndex]
                                val recentOtherModelsBuffer = recentOtherModelsBuffers[nodeIndex]
                                val iterTestFull = iterTestFulls[nodeIndex]
                                val behavior = behaviors[nodeIndex]

                                if (joiningLateRemainingIterations[nodeIndex] > 0) {
                                    joiningLateRemainingIterations[nodeIndex]--
                                    if (nodeIndex == 0) logger.debug { "JL => continue" }
                                    continue
                                }
                                if (nodeConfig.trainConfiguration.slowdown != Slowdowns.NONE) {
                                    if (slowdownRemainingIterations[nodeIndex] > 0) {
                                        slowdownRemainingIterations[nodeIndex]--
                                        if (nodeIndex == 0) logger.debug { "SD => continue" }
                                        continue
                                    } else {
                                        slowdownRemainingIterations[nodeIndex] =
                                            round(1 / nodeConfig.trainConfiguration.slowdown.multiplier).toInt() - 1
                                    }
                                }


                                if (iteration % iterationsBeforeSendings[nodeIndex] == 0) {
                                    if (behavior == Behaviors.BENIGN) {
                                        // Attack
                                        val attack = nodeConfig.modelPoisoningConfiguration.attack
                                        val attackVectors = attack.obj.generateAttack(
                                            nodeConfig.modelPoisoningConfiguration.numAttackers,
                                            oldParam,
                                            gradient,
                                            newOtherModelBuffer,
                                            random
                                        )
                                        newOtherModelBuffer.putAll(attackVectors)

                                        // Integrate parameters of other peers
                                        val numPeers = newOtherModelBuffer.size + 1
                                        val averageParams: INDArray
                                        if (numPeers == 1) {
                                            if (nodeIndex == 0) logger.debug { "No received params => skipping integration evaluation" }
                                            averageParams = newParams[nodeIndex]
                                            network.setParameters(averageParams)
                                        } else {
                                            if (nodeIndex == 0) logger.debug { "Params received => executing aggregation rule" }
                                            averageParams = nodeConfig.trainConfiguration.gar.obj.integrateParameters(
                                                network,
                                                oldParam,
                                                gradient,
                                                newOtherModelBuffer,
                                                recentOtherModelsBuffer,
                                                iterTestFull,
                                                countPerPeer,
                                                nodeIndex == 0
                                            )
                                            recentOtherModelsBuffer.addAll(newOtherModelBuffer.toList())
                                            while (recentOtherModelsBuffer.size > SIZE_RECENT_OTHER_MODELS) {
                                                recentOtherModelsBuffer.removeFirst()
                                            }
                                            network.setParameters(averageParams)
                                        }

                                        if (iteration % iterationsBeforeEvaluations[nodeIndex] == 0 && (nodeIndex == 0 || !ONLY_EVALUATE_FIRST_NODE)) {
                                            val elapsedTime2 = System.currentTimeMillis() - start
                                            val extraElements2 = mapOf(
                                                Pair("before or after averaging", "before"),
                                                Pair("#peers included in current batch", numPeers.toString())
                                            )
                                            evaluationProcessor.evaluate(
                                                iterTestFull,
                                                network,
                                                extraElements2,
                                                elapsedTime2,
                                                iteration,
                                                epoch,
                                                nodeIndex,
                                                nodeIndex == 0
                                            )
                                        }
                                    }
                                    newOtherModelBuffer.clear()
                                }

                                oldParams[nodeIndex] = network.params().dup()

                                epochEnd = fitNetwork(network, iterTrains[nodeIndex])

                                if (iteration % iterationsBeforeSendings[nodeIndex] == 0) {
                                    shareModel(
                                        network.params().dup(),
                                        nodeConfig.trainConfiguration,
                                        random,
                                        nodeIndex,
                                        newOtherModelBuffersTemp,
                                        countPerPeer
                                    )
                                }

                                if (iteration % iterationsBeforeEvaluations[nodeIndex] == 0 && (nodeIndex == 0 || !ONLY_EVALUATE_FIRST_NODE)) {
                                    val elapsedTime = System.currentTimeMillis() - start
                                    val extraElements = mapOf(
                                        Pair("before or after averaging", "after"),
                                        Pair("#peers included in current batch", "")
                                    )
                                    evaluationProcessor.evaluate(
                                        iterTestFull,
                                        network,
                                        extraElements,
                                        elapsedTime,
                                        iteration,
                                        epoch,
                                        nodeIndex,
                                        nodeIndex == 0
                                    )
                                }
                            }
                        }
                        logger.warn { "Test finished" }
                    }
                }
            } catch (e: Exception) {
                evaluationProcessor.error(e)
                e.printStackTrace()
            }
        }
    }

    private fun fitNetwork(network: MultiLayerNetwork, dataSetIterator: CustomDataSetIterator): Boolean {
        try {
            val ds = dataSetIterator.next()
            network.fit(ds)
        } catch (e: Exception) {
            return true
        }
        return false
    }

    private fun getSimilarPeers(
        trainDataSetIterator: CustomDataSetIterator,
        sraKeyPair: SRAKeyPair,
        toServerMessageBuffers: Array<CopyOnWriteArrayList<MsgPsiCaClientToServer>>,
        toClientMessageBuffers: Array<CopyOnWriteArrayList<MsgPsiCaServerToClient>>,
        i: Int,
    ): Map<Int, Int> {
        val encryptedLabels = clientsRequestsServerLabels(
            trainDataSetIterator.labels,
            sraKeyPair
        )
        toServerMessageBuffers
            .filterIndexed { index, _ -> index != i }
            .forEach { it.add(MsgPsiCaClientToServer(encryptedLabels, i)) }

        while (toServerMessageBuffers[i].size < toServerMessageBuffers.size - 1) {
            Thread.sleep(1)
        }

        for (toServerMessage in toServerMessageBuffers[i]) {
            val (reEncryptedLabels, filter) = serverRespondsClientRequests(
                trainDataSetIterator.labels,
                toServerMessage,
                sraKeyPair
            )
            val message = MsgPsiCaServerToClient(reEncryptedLabels, filter, i)
            toClientMessageBuffers[toServerMessage.client].add(message)
        }

        while (toClientMessageBuffers[i].size < toClientMessageBuffers.size - 1) {
            Thread.sleep(1)
        }

        return clientReceivesServerResponses(
            i,
            toClientMessageBuffers[i],
            sraKeyPair
        )
    }

    private fun shareModel(
        params: INDArray,
        trainConfiguration: TrainConfiguration,
        random: Random,
        nodeIndex: Int,
        newOtherModelBuffersTemp: Array<ConcurrentHashMap<Int, INDArray>>,
        countPerPeer: Map<Int, Int>
    ) {
        // Send to other peers
        val message = craftMessage(params, trainConfiguration.behavior, random)
        when (trainConfiguration.communicationPattern) {
            CommunicationPatterns.ALL -> newOtherModelBuffersTemp
                .filterIndexed { index, _ -> index != nodeIndex && index in countPerPeer.keys }
                .forEach { it[nodeIndex] = message }
            CommunicationPatterns.RANDOM -> newOtherModelBuffersTemp
                .filterIndexed { index, _ -> index != nodeIndex && index in countPerPeer.keys }
                .random()[nodeIndex] = message
            CommunicationPatterns.RR -> throw IllegalArgumentException("Not implemented yet")
            CommunicationPatterns.RING -> throw IllegalArgumentException("Not implemented yet")
        }
    }
}
