package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaClientToServer
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaServerToClient
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File
import java.nio.file.Paths
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread
import kotlin.random.Random

private val logger = KotlinLogging.logger("SimulatedRunner")

private const val NUM_RECENT_OTHER_MODELS = 20

class SimulatedRunner : Runner() {
    private val newOtherModelBuffers = ArrayList<ConcurrentHashMap<Int, INDArray>>()
    private val recentOtherModelsBuffers = ArrayList<ArrayDeque<Pair<Int, INDArray>>>()
    private val randoms = ArrayList<Random>()

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
            val (configs, figureNames) = generateConfigs(automation, automationPart)
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
                    for (i in testConfig.indices) {
                        newOtherModelBuffers.add(ConcurrentHashMap())
                        recentOtherModelsBuffers.add(ArrayDeque())
                        randoms.add(Random(i))
                        toServerMessageBuffers.add(CopyOnWriteArrayList())
                        toClientMessageBuffers.add(CopyOnWriteArrayList())
                        sraKeyPairs.add(SRAKeyPair.create(bigPrime, java.util.Random(i.toLong())))
                    }
                    logger.debug { "testConfig.indices: ${testConfig.indices}" }
                    val threads = ArrayList<Thread>()

                    for (simulationIndex in testConfig.indices) {
                        threads.add(thread {
                            val nodeConfig = testConfig[simulationIndex]
                            val dataset = nodeConfig.dataset
                            val datasetIteratorConfiguration = nodeConfig.datasetIteratorConfiguration
                            val behavior = nodeConfig.trainConfiguration.behavior
                            val (iterTrain, iterTrainFull, iterTest, iterTestFull) = getDataSetIterators(
                                dataset,
                                datasetIteratorConfiguration,
                                simulationIndex.toLong() * 10,
                                baseDirectory,
                                behavior
                            )
                            val network = generateNetwork(
                                dataset,
                                nodeConfig.nnConfiguration,
                                simulationIndex
                            )
                            if (simulationIndex == 0) {
                                network.setListeners(ScoreIterationListener(printScoreIterations))
                            }
                            val countPerPeer = getSimilarPeers(
                                iterTrain,
                                sraKeyPairs[simulationIndex],
                                toServerMessageBuffers,
                                toClientMessageBuffers,
                                simulationIndex
                            )

                            trainTestSendNetwork(
                                simulationIndex,
                                network,
                                simulationIndex == 0,

                                evaluationProcessor,
                                iterTestFull,

                                iterTrain,
                                iterTrainFull,
                                nodeConfig.trainConfiguration,
                                nodeConfig.modelPoisoningConfiguration,

                                iterTest,
                                countPerPeer
                            )
                        })
                    }
                    threads.forEach { it.join() }
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

    private fun trainTestSendNetwork(
        // General information
        simulationIndex: Int,
        network: MultiLayerNetwork,
        logging: Boolean,

        // Evaluation results
        evaluationProcessor: EvaluationProcessor,
        fullTestDataSetIterator: CustomDataSetIterator,

        // Training the network
        trainDataSetIterator: CustomDataSetIterator,
        fullTrainDataSetIterator: CustomDataSetIterator,
        trainConfiguration: TrainConfiguration,
        modelPoisoningConfiguration: ModelPoisoningConfiguration,

        // Integrating and distributing information to peers
        testDataSetIterator: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
    ) {
        val newOtherModels = newOtherModelBuffers[simulationIndex]
        if (trainConfiguration.joiningLate != TransmissionRounds.N0) {
            var count = 0
            val target = (countPerPeer.keys.size) * trainConfiguration.joiningLate.rounds
            while (count < target) {
                logger.debug { "Joining late: found $count of $target" }
                Thread.sleep(1000)
                count += newOtherModels.size
                newOtherModels.clear()
            }
            newOtherModels.clear()
        }
        val recentOtherModels = recentOtherModelsBuffers[simulationIndex]
        val random = randoms[simulationIndex]

        val gar = trainConfiguration.gar.obj
        var iterations = 0
        var iterationsToEvaluation = 0
        var iterationsToSending = 0
        var epoch = 0
        var slowDownStart: Long? = null
        val start = System.currentTimeMillis()

        epochLoop@ while (true) {
            if (logging) logger.debug { "Starting epoch: $epoch" }
            trainDataSetIterator.reset()
            fullTrainDataSetIterator.reset()
            var oldParams = network.params().dup()

            while (true) {
                if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                    iterationsToEvaluation = 0
                }
                if (iterationsToSending >= iterationsBeforeSending) {
                    iterationsToSending = 0
                }
                if (trainConfiguration.slowdown != Slowdowns.NONE) {
                    if (slowDownStart != null) {
                        val timePassed = System.currentTimeMillis() - slowDownStart
                        Thread.sleep((timePassed * (1.0 / trainConfiguration.slowdown.multiplier - 1)).toLong())
                    }
                    slowDownStart = System.currentTimeMillis()
                }

                // Train
                var endEpoch = false
                try {
                    network.fit(trainDataSetIterator.next())
                } catch (e: Exception) {
                    e.printStackTrace()
                    endEpoch = true
                }
                /*try { ONLY FOR BRISTLE
                    network.fit(fullTrainDataSetIterator.next())
                } catch (e: NoSuchElementException) {
                    fullTrainDataSetIterator.reset()
                    network.fit(fullTrainDataSetIterator.next())
                }
                val newParams = network.params().dup()
                val gradient = oldParams.sub(newParams)
                iterations++
                iterationsToEvaluation++
                iterationsToSending++

                if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                    // Test
                    if (logging) logger.debug { "Evaluating network " }
                    val elapsedTime = System.currentTimeMillis() - start
                    val extraElements = mapOf(
                        Pair("before or after averaging", "before"),
                        Pair("#peers included in current batch", "")
                    )
                    evaluationProcessor.evaluate(
                        fullTestDataSetIterator,
                        network,
                        extraElements,
                        elapsedTime,
                        iterations,
                        epoch,
                        simulationIndex,
                        logging)

                    if (iterationsToSending >= iterationsBeforeSending) {
                        // Attack
                        val attack = modelPoisoningConfiguration.attack
                        val attackVectors = attack.obj.generateAttack(
                            modelPoisoningConfiguration.numAttackers,
                            oldParams,
                            gradient,
                            newOtherModels,
                            random
                        )
                        newOtherModels.putAll(attackVectors)

                        // Integrate parameters of other peers
                        val numPeers = newOtherModels.size + 1
                        val averageParams: INDArray
                        if (numPeers == 1) {
                            if (logging) logger.debug { "No received params => skipping integration evaluation" }
                            averageParams = newParams
                            network.setParameters(averageParams)
                        } else {
                            if (logging) logger.debug { "Params received => executing aggregation rule" }
                            averageParams = gar.integrateParameters(
                                network,
                                oldParams,
                                gradient,
                                newOtherModels,
                                recentOtherModels,
                                fullTestDataSetIterator,
                                countPerPeer,
                                logging
                            )
                            recentOtherModels.addAll(newOtherModels.toList())
                            while (recentOtherModels.size > NUM_RECENT_OTHER_MODELS) {
                                recentOtherModels.removeFirst()
                            }
                            newOtherModels.clear()
                            network.setParameters(averageParams)
                            val elapsedTime2 = System.currentTimeMillis() - start
                            val extraElements2 = mapOf(
                                Pair("before or after averaging", "after"),
                                Pair("#peers included in current batch", numPeers.toString())
                            )
                            evaluationProcessor.evaluate(
                                fullTestDataSetIterator,
                                network,
                                extraElements2,
                                elapsedTime2,
                                iterations,
                                epoch,
                                simulationIndex,
                                logging
                            )
                        }

                        // Send to other peers
                        val message = craftMessage(averageParams.dup(), trainConfiguration.behavior, random)
                        when (trainConfiguration.communicationPattern) {
                            CommunicationPatterns.ALL -> newOtherModelBuffers
                                .filterIndexed { index, _ -> index != simulationIndex && index in countPerPeer.keys }
                                .forEach { it[simulationIndex] = message }
                            CommunicationPatterns.RANDOM -> newOtherModelBuffers
                                .filterIndexed { index, _ -> index != simulationIndex && index in countPerPeer.keys }
                                .random()[simulationIndex] = message
                            CommunicationPatterns.RR -> throw IllegalArgumentException("Not implemented yet")
                            CommunicationPatterns.RING -> throw IllegalArgumentException("Not implemented yet")
                        }
                    }

//                    if (iterationsToEvaluation >= iterationsBeforeEvaluation || (gar.isDirectIntegration() && newOtherModels.size > 0)) {
//                    }
                    // Send new parameters to other peers
//                        if (logging) logger.debug {
//                            "Sending model to peers: ${averageParams.getDouble(0)}, ${averageParams.getDouble(1)}, ${
//                                averageParams.getDouble(
//                                    2
//                                )
//                            }, ${averageParams.getDouble(3)}"
//                        }
                }
                oldParams = network.params().dup()
                if (iterations >= trainConfiguration.maxIteration.value) {
                    break@epochLoop
                }
                if (endEpoch) {
                    epoch++
                    break
                }*/
            }
        }
        logger.debug { "Done training the network" }
        evaluationProcessor.done()
    }
}
