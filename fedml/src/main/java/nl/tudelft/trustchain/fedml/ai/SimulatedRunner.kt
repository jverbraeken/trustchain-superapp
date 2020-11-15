package nl.tudelft.trustchain.fedml.ai

import com.google.common.hash.BloomFilter
import com.google.common.hash.Funnel
import com.google.common.hash.PrimitiveSink
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaClientToServer
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaServerToClient
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.math.BigInteger
import java.nio.file.Paths
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.CopyOnWriteArrayList
import java.util.stream.Collectors
import kotlin.concurrent.thread
import kotlin.random.Random

private val logger = KotlinLogging.logger("SimulatedRunner")

private const val NUM_RECENT_OTHER_MODELS: Int = 20

class SimulatedRunner : Runner() {
    private val newOtherModelBuffers = ArrayList<CopyOnWriteArrayList<INDArray>>()
    private val recentOtherModelsBuffers = ArrayList<ConcurrentLinkedDeque<INDArray>>()
    private val randoms = ArrayList<Random>()
    override val iterationsBeforeEvaluation = 100

    override fun run(
        baseDirectory: File,
        seed: Int,
        @Suppress("PARAMETER_NAME_CHANGED_ON_OVERRIDE") _unused: MLConfiguration
    ) {
        val config = loadConfig(baseDirectory)
        // All these things have to be initialized before any of the runner threads start
        val toServerMessageBuffers = ArrayList<CopyOnWriteArrayList<MsgPsiCaClientToServer>>()
        val toClientMessageBuffers = ArrayList<CopyOnWriteArrayList<MsgPsiCaServerToClient>>()
        val sraKeyPairs: MutableList<SRAKeyPair> = ArrayList()
        for (i in config.indices) {
            newOtherModelBuffers.add(CopyOnWriteArrayList())
            recentOtherModelsBuffers.add(ConcurrentLinkedDeque())
            randoms.add(Random(i))
            toServerMessageBuffers.add(CopyOnWriteArrayList())
            toClientMessageBuffers.add(CopyOnWriteArrayList())
            sraKeyPairs.add(SRAKeyPair.create(bigPrime, java.util.Random(i.toLong())))
        }
        for (i in config.indices) {
            thread {
                val trainDataSetIterator = getTrainDatasetIterator(
                    baseDirectory,
                    config[i].dataset,
                    config[i].datasetIteratorConfiguration,
                    config[i].trainConfiguration.behavior,
                    i
                )
                val testDataSetIterator = getTestDatasetIterator(
                    baseDirectory,
                    config[i].dataset,
                    config[i].datasetIteratorConfiguration,
                    config[i].trainConfiguration.behavior,
                    i
                )
                val evaluationProcessor = EvaluationProcessor(
                    baseDirectory,
                    "simulated",
                    config[i],
                    i,
                    listOf(
                        "before or after averaging",
                        "#peers included in current batch"
                    ),
                    "-simulation-$i"
                )
                val network = generateNetwork(
                    config[i].dataset,
                    config[i].nnConfiguration,
                    i
                )
                network.setListeners(ScoreIterationListener(printScoreIterations))




                val encryptedLabels = clientsRequestsServerLabels(
                    trainDataSetIterator.labels,
                    sraKeyPairs[i]
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
                        sraKeyPairs[i]
                    )
                    val message = MsgPsiCaServerToClient(reEncryptedLabels, filter, i)
                    toClientMessageBuffers[toServerMessage.client].add(message)
                }

                while (toClientMessageBuffers[i].size < toClientMessageBuffers.size - 1) {
                    Thread.sleep(1)
                }

                val similarPeers = clientReceivesServerResponses(
                    i,
                    toClientMessageBuffers[i],
                    sraKeyPairs[i]
                )




                trainTestSendNetwork(
                    i,
                    network,
                    evaluationProcessor,
                    trainDataSetIterator,
                    testDataSetIterator,
                    config[i].trainConfiguration,
                    config[i].modelPoisoningConfiguration,
                    similarPeers
                )
            }
        }
    }

    private fun trainTestSendNetwork(
        simulationIndex: Int,
        network: MultiLayerNetwork,
        evaluationProcessor: EvaluationProcessor,
        trainDataSetIterator: DataSetIterator,
        testDataSetIterator: DataSetIterator,
        trainConfiguration: TrainConfiguration,
        modelPoisoningConfiguration: ModelPoisoningConfiguration,
        similarPeers: List<Int>
    ) {
        val newOtherModels = newOtherModelBuffers[simulationIndex]
        val recentOtherModels = recentOtherModelsBuffers[simulationIndex]
        val random = randoms[simulationIndex]
        val batchSize = trainDataSetIterator.batch()
        val gar = trainConfiguration.gar.obj
        var iterations = 0
        var iterationsToEvaluation = 0
        for (epoch in 0 until trainConfiguration.numEpochs.value) {
            logger.debug { "Starting epoch: $epoch" }
            evaluationProcessor.epoch = epoch
            trainDataSetIterator.reset()
            val start = System.currentTimeMillis()
            var oldParams = network.params().dup()
            while (true) {

                // Train
                var endEpoch = false
                try {
                    network.fit(trainDataSetIterator.next())
                } catch (e: NoSuchElementException) {
                    endEpoch = true
                }
                val newParams = network.params().dup()
                val gradient = oldParams.sub(newParams)
                iterations += batchSize
                iterationsToEvaluation += batchSize

                if (iterationsToEvaluation >= iterationsBeforeEvaluation || (gar.isDirectIntegration() && newOtherModels.size > 0)) {

                    if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                        // Test
                        iterationsToEvaluation = 0
                        val end = System.currentTimeMillis()
                        logger.debug { "Evaluating network " }
                        evaluationProcessor.iteration = iterations
                        execEvaluationProcessor(
                            evaluationProcessor,
                            testDataSetIterator,
                            network,
                            EvaluationProcessor.EvaluationData(
                                "before", "", end - start, network.iterationCount, epoch
                            )
                        )
                    }

                    // Integrate parameters of other peers
                    val attack = modelPoisoningConfiguration.attack
                    val attackVectors = attack.obj.generateAttack(
                        modelPoisoningConfiguration.numAttackers,
                        oldParams,
                        gradient,
                        newOtherModels,
                        random
                    )
                    newOtherModels.addAll(attackVectors)
                    val numPeers = newOtherModels.size + 1
                    val averageParams: INDArray
                    if (numPeers == 1) {
                        logger.debug { "No received params => skipping integration evaluation" }
                        averageParams = newParams
                        network.setParameters(averageParams)
                    } else {
                        logger.debug { "Params received => executing aggregation rule" }

                        val start2 = System.currentTimeMillis()
                        averageParams = gar.integrateParameters(
                            oldParams,
                            gradient,
                            newOtherModels,
                            network,
                            testDataSetIterator,
                            recentOtherModels
                        )
                        recentOtherModels.addAll(newOtherModels)
                        while (recentOtherModels.size > NUM_RECENT_OTHER_MODELS) {
                            recentOtherModels.removeFirst()
                        }
                        newOtherModels.clear()
                        network.setParameters(averageParams)

                        if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                            val end2 = System.currentTimeMillis()
                            execEvaluationProcessor(
                                evaluationProcessor,
                                testDataSetIterator,
                                network,
                                EvaluationProcessor.EvaluationData(
                                    "after", numPeers.toString(), end2 - start2, iterations, epoch
                                )
                            )
                        }
                    }

                    // Send new parameters to other peers
                    val message = craftMessage(averageParams, trainConfiguration.behavior, random)
                    when (trainConfiguration.communicationPattern) {
                        CommunicationPatterns.ALL -> newOtherModelBuffers.filterIndexed { index, _ -> index != simulationIndex && index in similarPeers }
                            .forEach { it.add(message) }
                        CommunicationPatterns.RANDOM -> newOtherModelBuffers.filterIndexed { index, _ -> index != simulationIndex && index in similarPeers }
                            .random().add(message)
                        CommunicationPatterns.RR -> throw IllegalArgumentException("Not implemented yet")
                        CommunicationPatterns.RING -> throw IllegalArgumentException("Not implemented yet")
                    }
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

    private fun craftMessage(first: INDArray, behavior: Behaviors, random: Random): INDArray {
        return when (behavior) {
            Behaviors.BENIGN -> first
            Behaviors.NOISE -> craftNoiseMessage(first, random)
            Behaviors.LABEL_FLIP -> first
        }
    }

    private fun craftNoiseMessage(first: INDArray, random: Random): INDArray {
        val oldMatrix = first.toFloatMatrix()[0]
        val newMatrix = Array(1) { FloatArray(oldMatrix.size) }
        for (i in oldMatrix.indices) {
            newMatrix[0][i] = random.nextFloat() * 2 - 1
        }
        return NDArray(newMatrix)
    }

    private fun execEvaluationProcessor(
        evaluationProcessor: EvaluationProcessor,
        testDataSetIterator: DataSetIterator,
        network: MultiLayerNetwork,
        evaluationData: EvaluationProcessor.EvaluationData
    ) {
        testDataSetIterator.reset()
        evaluationProcessor.extraElements = mapOf(
            Pair("before or after averaging", evaluationData.beforeAfterAveraging),
            Pair("#peers included in current batch", evaluationData.numPeers)
        )
        evaluationProcessor.elapsedTime = evaluationData.elapsedTime
        val evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
        evaluationListener.callback = evaluationProcessor
        evaluationListener.iterationDone(
            network,
            evaluationData.iterationCount,
            evaluationData.epoch
        )
    }

    private fun loadConfig(baseDirectory: File): List<MLConfiguration> {
        val file = Paths.get(baseDirectory.path, "simulation.config").toFile()
        val lines = file.readLines()
        val configurations = arrayListOf<MLConfiguration>()

        var dataset: Datasets = Datasets.MNIST
        var optimizer: Optimizers = dataset.defaultOptimizer
        var learningRate: LearningRates = dataset.defaultLearningRate
        var momentum: Momentums = dataset.defaultMomentum
        var l2: L2Regularizations = dataset.defaultL2
        var batchSize: BatchSizes = dataset.defaultBatchSize
        var epoch: Epochs = Epochs.EPOCH_5
        var iteratorDistribution: IteratorDistributions = dataset.defaultIteratorDistribution
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
                            behavior = behavior
                        ),
                        ModelPoisoningConfiguration(
                            attack = modelPoisoningAttack,
                            numAttackers = numAttacker
                        )
                    )
                )
                continue
            }
            val split = line.split(' ')
            when (split[0]) {
                "dataset" -> dataset = loadDataset(split[1])
                "batchSize" -> batchSize = loadBatchSize(split[1])
                "iteratorDistribution" -> iteratorDistribution = loadIteratorDistribution(split[1])
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
}
