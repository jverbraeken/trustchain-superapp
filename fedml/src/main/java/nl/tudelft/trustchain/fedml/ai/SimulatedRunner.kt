package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaClientToServer
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaServerToClient
import nl.tudelft.trustchain.fedml.ui.Automation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
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

    fun automate(
        baseDirectory: File,
        automationFilename: String,
    ) {
        /*val trainDataSetIterators = (0 until 10).map {
            Datasets.MNIST.inst(
                DatasetIteratorConfiguration(BatchSizes.BATCH_5,
                    listOf(10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
                    *//*listOf(if (it == 0) 100 else 10,
                        if (it == 1) 100 else 10,
                        if (it == 2) 100 else 10,
                        if (it == 3) 100 else 10,
                        if (it == 4) 100 else 10,
                        if (it == 5) 100 else 10,
                        if (it == 6) 800 else 0,
                        if (it == 7) 800 else 0,
                        if (it == 8) 800 else 0,
                        if (it == 9) 800 else 0),*//*
                    MaxTestSamples.NUM_200),
                it.toLong(),
                CustomDataSetType.TRAIN,
                baseDirectory,
                Behaviors.BENIGN
            )
        }
        val testDataSetIterator = Datasets.MNIST.inst(
            DatasetIteratorConfiguration(BatchSizes.BATCH_5,
//                listOf(200, 200, 200, 200, 200, 200, 200, 200, 200, 200),
                listOf(100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                MaxTestSamples.NUM_100),
            0L,
            CustomDataSetType.FULL_TEST,
            baseDirectory,
            Behaviors.BENIGN
        )
        val globalNetworks = (0 until 11).map {
            generateNetwork(
                Datasets.MNIST,
                NNConfiguration(Datasets.MNIST.defaultOptimizer, Datasets.MNIST.defaultLearningRate, Datasets.MNIST.defaultMomentum, Datasets.MNIST.defaultL2),
                0
            )
        }
        logger.debug { "1" }
        val oldModel = globalNetworks[0].params().dup()
        (0 until 2).forEach { i ->
            logger.debug { "2" }
            repeat(1) {
                logger.debug { "3" }
                globalNetworks[i].fit(trainDataSetIterators[i])
            }
        }
        logger.debug { "3" }
        val newParams = globalNetworks[0].params().dup()
        val gradient = oldModel.sub(newParams)
        testDataSetIterator.reset()
        val evaluations = arrayOf(Evaluation())
        globalNetworks[0].doEvaluation(testDataSetIterator, *evaluations)
        for (evaluation in evaluations) {
            logger.debug { "${evaluation.javaClass.simpleName}:\n${evaluation.stats()}" }
        }
        logger.debug { "4" }
        val op1 = globalNetworks[0].params().add(globalNetworks[1].params()).div(2)
        globalNetworks[0].setParams(op1)
        logger.debug { "5" }

        testDataSetIterator.reset()
        val evaluations2 = arrayOf(Evaluation())
        globalNetworks[0].doEvaluation(testDataSetIterator, *evaluations2)
        for (evaluation in evaluations2) {
            logger.debug { "${evaluation.javaClass.simpleName}:\n${evaluation.stats()}" }
        }

        val op2 = Average().integrateParameters(globalNetworks[0], oldModel, gradient, listOf(globalNetworks[1]).mapIndexed { index, multiLayerNetwork -> Pair(index, multiLayerNetwork.params().dup()) }.toMap(), ArrayDeque(), testDataSetIterator, mapOf(), true)
        globalNetworks[0].setParams(op2)
        logger.debug { "6" }

        testDataSetIterator.reset()
        val evaluations3 = arrayOf(Evaluation())
        globalNetworks[0].doEvaluation(testDataSetIterator, *evaluations3)
        for (evaluation in evaluations3) {
            logger.debug { "${evaluation.javaClass.simpleName}:\n${evaluation.stats()}" }
        }
        print("hoi")*/

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

            val automation = loadAutomation(baseDirectory, automationFilename)
            val iterations = automation.fixedValues["iterations"]!!.toInt()
            val batchSize = loadBatchSize(automation.fixedValues["batchSize"]!!)
            val iteratorDistribution_ = automation.fixedValues["iteratorDistribution"]!!
            val iteratorDistribution = if (iteratorDistribution_.startsWith('[')) {
                iteratorDistribution_
                    .substring(1, iteratorDistribution_.length - 1)
                    .split(", ")
                    .map { it.toInt() }
            } else {
                loadIteratorDistribution(iteratorDistribution_).value
            }
            val maxTestSample = loadMaxTestSample(automation.fixedValues["maxTestSample"]!!)
            val optimizer = loadOptimizer(automation.fixedValues["optimizer"]!!)
            val learningRate = loadLearningRate(automation.fixedValues["learningRate"]!!)
            val momentum = loadMomentum(automation.fixedValues["momentum"]!!)
            val l2Regularization = loadL2Regularization(automation.fixedValues["l2Regularization"]!!)
            val epoch = loadEpoch(automation.fixedValues["epoch"]!!)
            val communicationPattern = loadCommunicationPattern(automation.fixedValues["communicationPattern"]!!)
            for (figure in automation.figures) {
                val figureName = figure.name
                val dataset = loadDataset(figure.fixedValues["dataset"]!!)
                val behavior = loadBehavior(figure.fixedValues["behavior"]!!)
                val modelPoisoningAttack = loadModelPoisoningAttack(figure.fixedValues["modelPoisoningAttack"]!!)
                val numNodes = figure.fixedValues["numNodes"]!!.toInt()
                val numAttackers = loadNumAttackers(figure.fixedValues["numAttackers"]!!)
                val firstNodeSpeed = figure.fixedValues["firstNodeSpeed"]?.toInt() ?: 0
                val firstNodeJoiningLate = figure.fixedValues["firstNodeJoiningLate"]?.equals("true") ?: false
                val overrideIteratorDistribution = figure.iteratorDistributions
                for (test in figure.tests) {
                    val gar = loadGAR(test.gar)
                    logger.error { "Going to test: $figureName - ${gar.text}" }
                    val configs = ArrayList<MLConfiguration>()
                    for (node in 0 until /*numNodes*/5) {
                        configs.add(
                            MLConfiguration(
                                dataset,
                                DatasetIteratorConfiguration(
                                    batchSize = batchSize,
                                    maxTestSamples = maxTestSample,
                                    distribution = overrideIteratorDistribution?.get(node % overrideIteratorDistribution.size)
                                        ?: iteratorDistribution
                                ),
                                NNConfiguration(
                                    optimizer = optimizer,
                                    learningRate = learningRate,
                                    momentum = momentum,
                                    l2 = l2Regularization
                                ),
                                TrainConfiguration(
                                    numEpochs = epoch,
                                    gar = gar,
                                    communicationPattern = communicationPattern,
                                    behavior = behavior,
                                    slowdown = if ((node == 0 && firstNodeSpeed == -1) || (node != 0 && firstNodeSpeed == 1)) Slowdowns.D2 else Slowdowns.NONE,
                                    joiningLate = if (node == 0 && firstNodeJoiningLate) TransmissionRounds.N2 else TransmissionRounds.N0
                                ),
                                ModelPoisoningConfiguration(
                                    attack = modelPoisoningAttack,
                                    numAttackers = numAttackers
                                )
                            )
                        )
                    }
                    evaluationProcessor.newSimulation(figureName + " - " + gar.id, configs)
                    // All these things have to be initialized before any of the runner threads start
                    val toServerMessageBuffers = ArrayList<CopyOnWriteArrayList<MsgPsiCaClientToServer>>()
                    val toClientMessageBuffers = ArrayList<CopyOnWriteArrayList<MsgPsiCaServerToClient>>()
                    val sraKeyPairs = ArrayList<SRAKeyPair>()
                    for (i in configs.indices) {
                        newOtherModelBuffers.add(ConcurrentHashMap())
                        recentOtherModelsBuffers.add(ArrayDeque())
                        randoms.add(Random(i))
                        toServerMessageBuffers.add(CopyOnWriteArrayList())
                        toClientMessageBuffers.add(CopyOnWriteArrayList())
                        sraKeyPairs.add(SRAKeyPair.create(bigPrime, java.util.Random(i.toLong())))
                    }
                    logger.debug { "config.indices: ${configs.indices}" }
                    val threads = ArrayList<Thread>()
                    for (simulationIndex in configs.indices) {
                        threads.add(thread {
                            val config = configs[simulationIndex]
                            val dataset = config.dataset
                            val datasetIteratorConfiguration = config.datasetIteratorConfiguration
                            val behavior = config.trainConfiguration.behavior
                            val (iterTrain, iterTrainFull, iterTest, iterTestFull) = getDataSetIterators(
                                dataset,
                                datasetIteratorConfiguration,
                                simulationIndex.toLong(),
                                baseDirectory,
                                behavior
                            )
                            val network = generateNetwork(
                                dataset,
                                config.nnConfiguration,
                                simulationIndex
                            )

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
                                iterations,

                                evaluationProcessor,
                                iterTestFull,

                                iterTrain,
                                iterTrainFull,
                                configs[simulationIndex].trainConfiguration,
                                configs[simulationIndex].modelPoisoningConfiguration,

                                iterTest,
                                countPerPeer
                            )
                        })
                    }
                    threads.forEach { it.join() }
                    logger.warn { "Test finished, iterations=$iterations" }
                }
            }
        }
    }

    override fun run(
        baseDirectory: File,
        seed: Int,
        @Suppress("PARAMETER_NAME_CHANGED_ON_OVERRIDE") _unused: MLConfiguration,
    ) {
        var configs = loadConfig(baseDirectory)
        // All these things have to be initialized before any of the runner threads start
        val toServerMessageBuffers = ArrayList<CopyOnWriteArrayList<MsgPsiCaClientToServer>>()
        val toClientMessageBuffers = ArrayList<CopyOnWriteArrayList<MsgPsiCaServerToClient>>()
        val sraKeyPairs = ArrayList<SRAKeyPair>()
        configs = configs.subList(0, 10)
        for (i in configs.indices) {
            newOtherModelBuffers.add(ConcurrentHashMap())
            recentOtherModelsBuffers.add(ArrayDeque())
            randoms.add(Random(i))
            toServerMessageBuffers.add(CopyOnWriteArrayList())
            toClientMessageBuffers.add(CopyOnWriteArrayList())
            sraKeyPairs.add(SRAKeyPair.create(bigPrime, java.util.Random(i.toLong())))
        }
        logger.debug { "config.indices: ${configs.indices}" }
        val evaluationProcessor = EvaluationProcessor(
            baseDirectory,
            "simulated",
            listOf(
                "before or after averaging",
                "#peers included in current batch"
            )
        )
        evaluationProcessor.newSimulation("simulation", configs)

        val threads = ArrayList<Thread>()
        for (simulationIndex in configs.indices) {
            threads.add(thread {
                val config = configs[simulationIndex]
                val dataset = config.dataset
                val datasetIteratorConfiguration = config.datasetIteratorConfiguration
                val behavior = config.trainConfiguration.behavior
                val (iterTrain, iterTrainFull, iterTest, iterTestFull) = getDataSetIterators(
                    dataset,
                    datasetIteratorConfiguration,
                    simulationIndex.toLong(),
                    baseDirectory,
                    behavior
                )
                val network = generateNetwork(
                    dataset,
                    config.nnConfiguration,
                    simulationIndex
                )

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
                    -1,

                    evaluationProcessor,
                    iterTestFull,

                    iterTrain,
                    iterTrainFull,
                    configs[simulationIndex].trainConfiguration,
                    configs[simulationIndex].modelPoisoningConfiguration,

                    iterTest,
                    countPerPeer,
                )
            })
        }
    }

    private fun loadAutomation(baseDirectory: File, automationFilename: String): Automation {
        val file = Paths.get(baseDirectory.path, "automation/$automationFilename.config").toFile()
        val string = file.readLines().joinToString("")
        return Json.decodeFromString(string)
    }

    private fun loadConfig(baseDirectory: File): List<MLConfiguration> {
        val file = Paths.get(baseDirectory.path, "automation/simulation.config").toFile()
        val lines = file.readLines()
        val configurations = arrayListOf<MLConfiguration>()

        var dataset = Datasets.MNIST
        var optimizer = dataset.defaultOptimizer
        var learningRate = dataset.defaultLearningRate
        var momentum = dataset.defaultMomentum
        var l2 = dataset.defaultL2
        var batchSize = dataset.defaultBatchSize
        var epoch = Epochs.EPOCH_5
        var iteratorDistribution = dataset.defaultIteratorDistribution.value
        var maxTestSample = MaxTestSamples.NUM_40
        var gar = GARs.BRISTLE
        var communicationPattern = CommunicationPatterns.RANDOM
        var behavior = Behaviors.BENIGN
        var modelPoisoningAttack = ModelPoisoningAttacks.NONE
        var numAttacker = NumAttackers.NUM_2

        for (line in lines) {
            if (line == "## end") {
                configurations.add(
                    MLConfiguration(
                        dataset,
                        DatasetIteratorConfiguration(
                            batchSize = batchSize,
                            maxTestSamples = maxTestSample,
                            distribution = iteratorDistribution
                        ),
                        NNConfiguration(
                            optimizer = optimizer,
                            learningRate = learningRate,
                            momentum = momentum,
                            l2 = l2
                        ),
                        TrainConfiguration(
                            numEpochs = epoch,
                            gar = gar,
                            communicationPattern = communicationPattern,
                            behavior = behavior,
                            Slowdowns.NONE,
                            TransmissionRounds.N0
                        ),
                        ModelPoisoningConfiguration(
                            attack = modelPoisoningAttack,
                            numAttackers = numAttacker
                        )
                    )
                )
                continue
            }
            val split = line.split(' ', limit = 2)
            when (split[0]) {
                "dataset" -> dataset = loadDataset(split[1])
                "batchSize" -> batchSize = loadBatchSize(split[1])
                "iteratorDistribution" -> {
                    iteratorDistribution = if (split[1].startsWith('[')) {
                        split[1]
                            .substring(1, split[1].length - 1)
                            .split(", ")
                            .map { it.toInt() }
                    } else {
                        loadIteratorDistribution(split[1]).value
                    }
                }
                "maxTestSample" -> maxTestSample = loadMaxTestSample(split[1])
                "optimizer" -> optimizer = loadOptimizer(split[1])
                "learningRate" -> learningRate = loadLearningRate(split[1])
                "momentum" -> momentum = loadMomentum(split[1])
                "l2Regularization" -> l2 = loadL2Regularization(split[1])
                "epoch" -> epoch = loadEpoch(split[1])
                "gar" -> gar = loadGAR(split[1])
                "communicationPattern" -> communicationPattern = loadCommunicationPattern(split[1])
                "behavior" -> behavior = loadBehavior(split[1])
                "modelPoisoningAttack" -> modelPoisoningAttack = loadModelPoisoningAttack(split[1])
                "numAttackers" -> numAttacker = loadNumAttackers(split[1])
            }
        }
        return configurations
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
        stopIterations: Int,

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
        if (logging) {
            network.setListeners(ScoreIterationListener(printScoreIterations))
        }
        /*var iterationsToEvaluation = 0
        for (i in 0 until trainConfiguration.numEpochs.value) {
            trainDataSetIterator.reset()
            val start = System.currentTimeMillis()
            while (true) {
                var endEpoch = false
                try {
                    network.fit(trainDataSetIterator.next())
                } catch (e: NoSuchElementException) {
                    endEpoch = true
                }
                iterationsToEvaluation += trainDataSetIterator.batch()

                if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                    iterationsToEvaluation = 0
                    val end = System.currentTimeMillis()
                    evaluationProcessor.evaluate(
                        testDataSetIterator,
                        network,
                        mapOf(),
                        end - start,
                        0,
                        0,
                        0,
                        true
                    )
                }
                if (endEpoch) {
                    break
                }
            }
        }*/

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
        val batchSize = trainDataSetIterator.batch()
        val gar = trainConfiguration.gar.obj
        var iterations = 0
        var iterationsToEvaluation = 0
        var iterationsToSending = 0
        var slowDownStart: Long? = null
        for (epoch in 0 until trainConfiguration.numEpochs.value) {
            if (logging) logger.debug { "Starting epoch: $epoch" }
            trainDataSetIterator.reset()
            fullTrainDataSetIterator.reset()
            val start = System.currentTimeMillis()
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
                } catch (e: NoSuchElementException) {
                    endEpoch = true
                }
//                try { only for bristle
//                    network.fit(fullTrainDataSetIterator.next())
//                } catch (e: NoSuchElementException) {
//                    fullTrainDataSetIterator.reset()
//                    network.fit(fullTrainDataSetIterator.next())
//                }
                val newParams = network.params().dup()
                val gradient = oldParams.sub(newParams)
                iterations += 1
                iterationsToEvaluation += 1
                iterationsToSending += 1



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
                if (stopIterations >= 0 && iterations >= stopIterations) {
                    return
                }
                oldParams = network.params().dup()
                if (endEpoch) {
                    break
                }
            }
        }
        logger.debug { "Done training the network" }
        evaluationProcessor.done()
    }
}
