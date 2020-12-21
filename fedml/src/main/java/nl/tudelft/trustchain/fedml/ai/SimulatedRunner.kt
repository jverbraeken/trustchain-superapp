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

private const val NUM_RECENT_OTHER_MODELS = 20

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
            val automation = loadAutomation(baseDirectory)
            logger.debug { "Automation loaded" }
            val (configs, figureNames) = generateConfigs(automation, automationPart)
            logger.debug { "Configs generated" }
            val figures = automation.figures
            val myFigures =
                if (automationPart == 0)
                    figures.subList(0, figures.size / 2)
                else
                    figures.subList(figures.size / 2, figures.size)

            for (figure in myFigures.indices) {
                val figureName = figureNames[figure]
                val figureConfig = configs[figure]

                for (test in figureConfig.indices) {
                    val testConfig = figureConfig[test]
                    logger.error { "Going to test: $figureName - ${testConfig[0].trainConfiguration.gar.id}" }
                    evaluationProcessor.newSimulation("$figureName - ${testConfig[0].trainConfiguration.gar.id}", testConfig)
                    // All these things have to be initialized before any of the runner threads start
                    val toServerMessageBuffers = ArrayList<CopyOnWriteArrayList<MsgPsiCaClientToServer>>()
                    val toClientMessageBuffers = ArrayList<CopyOnWriteArrayList<MsgPsiCaServerToClient>>()
                    val sraKeyPairs = ArrayList<SRAKeyPair>()
                    val randoms = arrayListOf<Random>()
                    val newOtherModelBuffers = arrayListOf<ConcurrentHashMap<Int, INDArray>>()
                    val recentOtherModelsBuffers = arrayListOf<ArrayDeque<Pair<Int, INDArray>>>()
                    for (i in testConfig.indices) {
                        newOtherModelBuffers.add(ConcurrentHashMap())
                        recentOtherModelsBuffers.add(ArrayDeque())
                        randoms.add(Random(i))
                        toServerMessageBuffers.add(CopyOnWriteArrayList())
                        toClientMessageBuffers.add(CopyOnWriteArrayList())
                        sraKeyPairs.add(SRAKeyPair.create(bigPrime, java.util.Random(i.toLong())))
                    }
                    logger.debug { "testConfig.indices: ${testConfig.indices}" }
                    val nodeConfig = arrayListOf<MLConfiguration>()
                    val dataset = arrayListOf<Datasets>()
                    val datasetIteratorConfiguration = arrayListOf<DatasetIteratorConfiguration>()
                    val behavior = arrayListOf<Behaviors>()
                    val iterTrain = arrayListOf<CustomDataSetIterator>()
                    val iterTrainFull = arrayListOf<CustomDataSetIterator>()
                    val iterTest = arrayListOf<CustomDataSetIterator>()
                    val iterTestFull = arrayListOf<CustomDataSetIterator>()
                    val network = arrayListOf<MultiLayerNetwork>()
                    val countPerPeer = arrayListOf<Map<Int, Int>>()
                    val joiningLateRemainingIterations = arrayListOf<Int>()
                    val slowdownRemainingIterations = arrayListOf<Int>()
                    val oldParams = arrayListOf<INDArray>()

                    val threads = ArrayList<Thread>()
                    for (simulationIndex in testConfig.indices) {
                        nodeConfig.add(testConfig[simulationIndex])
                        dataset.add(nodeConfig.last().dataset)
                        datasetIteratorConfiguration.add(nodeConfig.last().datasetIteratorConfiguration)
                        behavior.add(nodeConfig.last().trainConfiguration.behavior)
                        val iters = getDataSetIterators(
                            dataset.last(),
                            datasetIteratorConfiguration.last(),
                            simulationIndex.toLong() * 10,
                            baseDirectory,
                            behavior.last()
                        )
                        iterTrain.add(iters[0])
                        iterTrainFull.add(iters[1])
                        iterTest.add(iters[2])
                        iterTestFull.add(iters[3])
                        network.add(generateNetwork(
                            dataset.last(),
                            nodeConfig.last().nnConfiguration,
                            simulationIndex
                        ))
                        if (simulationIndex == 0) {
                            network.last().setListeners(ScoreIterationListener(printScoreIterations))
                        }
                        joiningLateRemainingIterations.add(
                            nodeConfig[simulationIndex].trainConfiguration.joiningLate.rounds * iterationsBeforeSending
                        )
                        slowdownRemainingIterations.add(0)
                        oldParams.add(NDArray())
                        threads.add(thread {
                            countPerPeer.add(getSimilarPeers(
                                iterTrain.last(),
                                sraKeyPairs[simulationIndex],
                                toServerMessageBuffers,
                                toClientMessageBuffers,
                                simulationIndex
                            ))
                        })
                    }
                    threads.forEach { it.join() }

                    var epochEnd = true
                    var epoch = -1
                    val start = System.currentTimeMillis()
                    for (iteration in 0 until nodeConfig[0].trainConfiguration.maxIteration.value) {
                        if (epochEnd) {
                            epoch++
                            logger.debug { "Epoch: $epoch" }
                            epochEnd = false
                            iterTrain.forEach { it.reset() }
                            iterTrainFull.forEach { it.reset() }
                            network.forEachIndexed { i, n ->
                                oldParams[i] = n.params().dup()
                            }
                        }
                        logger.debug { "Iteration: $iteration" }
                        for (simulationIndex in testConfig.indices) {
                            logger.debug { "Node: $simulationIndex" }
                            if (joiningLateRemainingIterations[simulationIndex] > 0) {
                                joiningLateRemainingIterations[simulationIndex]--
                                logger.debug { "JL => continue" }
                                continue
                            }
                            if (nodeConfig[0].trainConfiguration.slowdown != Slowdowns.NONE) {
                                if (slowdownRemainingIterations[simulationIndex] > 0) {
                                    slowdownRemainingIterations[simulationIndex]--
                                    logger.debug { "SD => continue" }
                                    continue
                                } else {
                                    slowdownRemainingIterations[simulationIndex] =
                                        round(1 / nodeConfig[0].trainConfiguration.slowdown.multiplier).toInt() - 1
                                }
                            }

                            try {
                                network[simulationIndex].fit(iterTrain[simulationIndex].next())
                            } catch (e: Exception) {
                                epochEnd = true
                            }
                            val newParams = network[simulationIndex].params().dup()
                            val gradient = oldParams[simulationIndex].sub(newParams)

                            if (iteration % iterationsBeforeSending == 0) {
                                // Testing
                                if (simulationIndex == 0) logger.debug { "Evaluating network " }
                                val elapsedTime = System.currentTimeMillis() - start
                                val extraElements = mapOf(
                                    Pair("before or after averaging", "before"),
                                    Pair("#peers included in current batch", "")
                                )
                                evaluationProcessor.evaluate(
                                    iterTestFull[simulationIndex],
                                    network[simulationIndex],
                                    extraElements,
                                    elapsedTime,
                                    iteration,
                                    epoch,
                                    simulationIndex,
                                    simulationIndex == 0)

                                // Attack
                                val attack = nodeConfig[simulationIndex].modelPoisoningConfiguration.attack
                                val attackVectors = attack.obj.generateAttack(
                                    nodeConfig[simulationIndex].modelPoisoningConfiguration.numAttackers,
                                    oldParams[simulationIndex],
                                    gradient,
                                    newOtherModelBuffers[simulationIndex],
                                    randoms[simulationIndex]
                                )
                                newOtherModelBuffers[simulationIndex].putAll(attackVectors)

                                // Integrate parameters of other peers
                                val numPeers = newOtherModelBuffers[simulationIndex].size + 1
                                val averageParams: INDArray
                                if (numPeers == 1) {
                                    if (simulationIndex == 0) logger.debug { "No received params => skipping integration evaluation" }
                                    averageParams = newParams
                                    network[simulationIndex].setParameters(averageParams)
                                } else {
                                    if (simulationIndex == 0) logger.debug { "Params received => executing aggregation rule" }
                                    averageParams = nodeConfig[simulationIndex].trainConfiguration.gar.obj.integrateParameters(
                                        network[simulationIndex],
                                        oldParams[simulationIndex],
                                        gradient,
                                        newOtherModelBuffers[simulationIndex],
                                        recentOtherModelsBuffers[simulationIndex],
                                        iterTestFull[simulationIndex],
                                        countPerPeer[simulationIndex],
                                        simulationIndex == 0
                                    )
                                    recentOtherModelsBuffers[simulationIndex].addAll(newOtherModelBuffers[simulationIndex].toList())
                                    while (recentOtherModelsBuffers[simulationIndex].size > NUM_RECENT_OTHER_MODELS) {
                                        recentOtherModelsBuffers[simulationIndex].removeFirst()
                                    }
                                    newOtherModelBuffers[simulationIndex].clear()
                                    network[simulationIndex].setParameters(averageParams)
                                    val elapsedTime2 = System.currentTimeMillis() - start
                                    val extraElements2 = mapOf(
                                        Pair("before or after averaging", "after"),
                                        Pair("#peers included in current batch", numPeers.toString())
                                    )
                                    evaluationProcessor.evaluate(
                                        iterTestFull[simulationIndex],
                                        network[simulationIndex],
                                        extraElements2,
                                        elapsedTime2,
                                        iteration,
                                        epoch,
                                        simulationIndex,
                                        simulationIndex == 0
                                    )
                                }

                                // Send to other peers
                                val message = craftMessage(averageParams.dup(), nodeConfig[simulationIndex].trainConfiguration.behavior, randoms[simulationIndex])
                                when (nodeConfig[simulationIndex].trainConfiguration.communicationPattern) {
                                    CommunicationPatterns.ALL -> newOtherModelBuffers
                                        .filterIndexed { index, _ -> index != simulationIndex && index in countPerPeer[simulationIndex].keys }
                                        .forEach { it[simulationIndex] = message }
                                    CommunicationPatterns.RANDOM -> newOtherModelBuffers
                                        .filterIndexed { index, _ -> index != simulationIndex && index in countPerPeer[simulationIndex].keys }
                                        .random()[simulationIndex] = message
                                    CommunicationPatterns.RR -> throw IllegalArgumentException("Not implemented yet")
                                    CommunicationPatterns.RING -> throw IllegalArgumentException("Not implemented yet")
                                }
                            }
                        }
                    }
                    logger.warn { "Test finished" }
                }
            }
        }
    }

    private fun getSimilarPeers(
        trainDataSetIterator: CustomDataSetIterator,
        sraKeyPair: SRAKeyPair,
        toServerMessageBuffers: ArrayList<CopyOnWriteArrayList<MsgPsiCaClientToServer>>,
        toClientMessageBuffers: ArrayList<CopyOnWriteArrayList<MsgPsiCaServerToClient>>,
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
}
