package nl.tudelft.trustchain.fedml.ai

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaClientToServer
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaServerToClient
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.lang.IllegalStateException
import java.nio.file.Paths
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.collections.ArrayDeque
import kotlin.collections.ArrayList
import kotlin.concurrent.thread
import kotlin.random.Random

private val logger = KotlinLogging.logger("SimulatedRunner")

private const val NUM_RECENT_OTHER_MODELS = 20

class SimulatedRunner : Runner() {
    private val newOtherModelBuffers = ArrayList<ConcurrentHashMap<Int, INDArray>>()
    private val recentOtherModelsBuffers = ArrayList<ArrayDeque<Pair<Int, INDArray>>>()
    private val randoms = ArrayList<Random>()
    override val iterationsBeforeEvaluation = 100

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
        configs = configs.subList(0, 1)
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
            configs,
            listOf(
                "before or after averaging",
                "#peers included in current batch"
            )
        )
        val globalNetworks = (0 until 10).map {
            generateNetwork(
                configs[0].dataset,
                configs[0].nnConfiguration,
                0
            )
        }
        val trainDataSetIterators = (0 until 10).map {
            configs[0].dataset.inst(
                DatasetIteratorConfiguration(BatchSizes.BATCH_5,
                    listOf(if (it == 0) 800 else 0,
                        if (it == 1) 800 else 0,
                        if (it == 2) 800 else 0,
                        if (it == 3) 800 else 0,
                        if (it == 4) 800 else 0,
                        if (it == 5) 800 else 0,
                        if (it == 6) 800 else 0,
                        if (it == 7) 800 else 0,
                        if (it == 8) 800 else 0,
                        if (it == 9) 800 else 0),
                    MaxTestSamples.NUM_200),
                0L,
                CustomDataSetType.TRAIN,
                baseDirectory,
                Behaviors.BENIGN
            )
        }
        val ewcTrainDataSetIterator =
            configs[0].dataset.inst(
                DatasetIteratorConfiguration(BatchSizes.BATCH_5,
                    listOf(800, 800, 800, 800, 800, 800, 800, 800, 800, 800),
                    MaxTestSamples.NUM_200),
                0L,
                CustomDataSetType.TRAIN,
                baseDirectory,
                Behaviors.BENIGN
            )
        val testDataSetIterator = configs[0].dataset.inst(
            DatasetIteratorConfiguration(BatchSizes.BATCH_5,
                listOf(200, 200, 200, 200, 200, 200, 200, 200, 200, 200),
                MaxTestSamples.NUM_200),
            0L,
            CustomDataSetType.FULL_TEST,
            baseDirectory,
            Behaviors.BENIGN
        )






        LossEWC.model = globalNetworks[0]
        repeat(50) {
            repeat(10) { task ->
                /*val precisionMatrices = globalNetworks[0]
                    .paramTable()
                    .map { (key, value) ->
                        Pair(
                            key,
                            NDArray(value.shape().map { dimension -> dimension.toInt() }.toIntArray()))
                    }
                    .toMap()
                val oldTasks = (1 until task).map { subTask ->
                    (0 until 9).map {
                        try {
                            trainDataSetIterators[subTask].next()  //
                        } catch (e: NoSuchElementException) {
                            trainDataSetIterators[subTask].reset()  //
                            trainDataSetIterators[subTask].next()  //
                        }
                    }
                }.flatten()
                var count = 0
                for (dataset in oldTasks) {
                    for (i in 0 until dataset.features.size(0)) {
                        count++
                        globalNetworks[0].input = dataset.features.slice(i).reshape(1, 784)
                        globalNetworks[0].labels = dataset.labels.slice(i).reshape(1, 10)
                        globalNetworks[0].computeGradient()
                        val gradient = globalNetworks[0].gradient()
                        val gradientForVariable = gradient.gradientForVariable()
                        precisionMatrices.forEach { matrix ->
                            matrix.value.addi(gradientForVariable[matrix.key]!!.mul(gradientForVariable[matrix.key]))
                        }
                    }
                }
                precisionMatrices.forEach { matrix -> matrix.value.divi(count) }
                LossEWC.precisionMatrices = precisionMatrices
                LossEWC.model = globalNetworks[0]
                LossEWC.means = globalNetworks[0].paramTable().map{(k, v) -> Pair(k, v.dup())}.toMap()*/





                repeat(75) {
                    val elem = try {
                        trainDataSetIterators[task].next()  //
                    } catch (e: NoSuchElementException) {
                        trainDataSetIterators[task].reset()  //
                        trainDataSetIterators[task].next()  //
                    }
                    globalNetworks[0].fit(elem)
                }
                logger.error { "Finished index: $task" }


                // consolidate
                LossEWC.old_var_list = globalNetworks[0].paramTable().map { (n, p) -> Pair(n, p.dup()) }.toMap()
                val grads: MutableMap<String, INDArray> = globalNetworks[0]
                    .paramTable()
                    .map { (key, value) ->
                        Pair(
                            key,
                            NDArray(value.shape().map { dimension -> dimension.toInt() }.toIntArray()))
                    }
                    .toMap()
                    .toMutableMap()
                repeat(75) {
                    val batch = try {
                        trainDataSetIterators[task].next()
                    } catch (e: IllegalStateException) {
                        trainDataSetIterators[task].reset()
                        trainDataSetIterators[task].next()
                    }
                    for (i in 0 until batch.features.size(0)) {
                        val input = batch.features.slice(i).reshape(1, 28 * 28)
                        val label = batch.labels.slice(i).reshape(1, 10)
                        grads.putAll(globalNetworks[0].calculateGradients(
                            input,
                            label,
                            null,
                            null
                        ).first.gradientForVariable())
                    }
                }
                for (entry in grads) {
                    grads[entry.key] = entry.value.mul(entry.value)
                }



                LossEWC.fishers = {
                    val gradientForVariable: MutableMap<String, INDArray> = globalNetworks[0]
                        .paramTable()
                        .map { (key, value) ->
                            Pair(
                                key,
                                NDArray(value.shape().map { dimension -> dimension.toInt() }.toIntArray()))
                        }
                        .toMap()
                        .toMutableMap()

                    for (i in 0 until a.features.size(0)) {
//                    globalNetworks[0].input = a.features.slice(i).reshape(1, 784)
//                    globalNetworks[0].labels = a.labels.slice(i).reshape(1, 10)
                        val gradientAndAsdf = globalNetworks[0].calculateGradients(
                            a.features.slice(i).reshape(1, 28 * 28),
                            a.labels.slice(i).reshape(1, 10),
                            null,
                            null
                        )
                        gradientForVariable.putAll(gradientAndAsdf.first.gradientForVariable())
                    }
                    gradientForVariable.forEach { it.value.mul(it.value).div(a.features.size(0)) }
                    gradientForVariable
                }.invoke()
                evaluationProcessor.iteration = 0
                execEvaluationProcessor(
                    evaluationProcessor,
                    testDataSetIterator,
                    globalNetworks[0],
                    EvaluationProcessor.EvaluationData(
                        "before", "", 0, globalNetworks[0].iterationCount, 0
                    ),
                    0,
                    true
                )
            }
        }










        repeat(50)
        {
            repeat(20) {
                globalNetworks.zip(trainDataSetIterators).forEach {
                    val elem = try {
                        it.second.next()
                    } catch (e: IllegalStateException) {
                        it.second.reset()
                        it.second.next()
                    }
                    it.first.fit(elem)

                    val precision_matrices = it.first
                        .paramTable()
                        .map { (key, value) ->
                            Pair(
                                key,
                                NDArray(value.shape().map { dimension -> dimension.toInt() }.toIntArray()))
                        }
                        .toMap()
                    for (i in 0 until elem.features.size(0)) {
                        val asdf = elem.features.shape().map { it.toInt() }.toIntArray()
                        NDArray(asdf)
                        it.first.input = elem.features.slice(i).reshape(1, 784)
                        it.first.labels = elem.labels.slice(i).reshape(1, 10)
                        it.first.computeGradientAndScore()
                        val gradient = it.first.gradient()
                        val gradientForVariable = gradient.gradientForVariable()
                        precision_matrices.forEach { matrix ->
                            matrix.value.addi(gradientForVariable[matrix.key]!!.mul(gradientForVariable[matrix.key]))
                        }
                    }
                    precision_matrices.forEach { matrix -> matrix.value.divi(elem.features.size(0)) }
                }
            }
            var newParams = globalNetworks[0].params().dup()
            (1 until 10).forEach {
                newParams = newParams.add(globalNetworks[it].params().dup())
            }
            newParams = newParams.div(10)
            globalNetworks.forEach { it.setParams(newParams.dup()) }
            evaluationProcessor.iteration = 0
            execEvaluationProcessor(
                evaluationProcessor,
                testDataSetIterator,
                globalNetworks[0],
                EvaluationProcessor.EvaluationData(
                    "before", "", 0, globalNetworks[0].iterationCount, 0
                ),
                0,
                true
            )
        }







        for (simulationIndex in configs.indices) {
            thread {
                val config = configs[simulationIndex]
                val dataset = config.dataset
                val datasetIteratorConfiguration = config.datasetIteratorConfiguration
                val behavior = config.trainConfiguration.behavior
                val trainDataSetIterator = dataset.inst(
                    datasetIteratorConfiguration,
                    simulationIndex.toLong(),
                    CustomDataSetType.TRAIN,
                    baseDirectory,
                    behavior
                )
                val testDataSetIterator = dataset.inst(
                    datasetIteratorConfiguration,
                    simulationIndex.toLong(),
                    CustomDataSetType.TEST,
                    baseDirectory,
                    behavior
                )
                val fullTestDataSetIterator = dataset.inst(
                    datasetIteratorConfiguration,
                    simulationIndex.toLong(),
                    CustomDataSetType.FULL_TEST,
                    baseDirectory,
                    behavior
                )
                /*val network = generateNetwork(
                    dataset,
                    config.nnConfiguration,
                    simulationIndex
                )
                network.setParams(globalNetwork.params().dup())
                evaluationProcessor.iteration = 0
                execEvaluationProcessor(
                    evaluationProcessor,
                    fullTestDataSetIterator,
                    network,
                    EvaluationProcessor.EvaluationData(
                        "before", "", 0, globalNetwork.iterationCount, 0
                    ),
                    0,
                    true
                )*/

                val countPerPeer = getSimilarPeers(
                    trainDataSetIterator,
                    sraKeyPairs[simulationIndex],
                    toServerMessageBuffers,
                    toClientMessageBuffers,
                    simulationIndex
                )

                /*trainTestSendNetwork(
                    simulationIndex,
                    network,
                    simulationIndex == 0,

                    evaluationProcessor,
                    fullTestDataSetIterator,

                    trainDataSetIterator,
                    configs[simulationIndex].trainConfiguration,
                    configs[simulationIndex].modelPoisoningConfiguration,

                    testDataSetIterator,
                    countPerPeer,
                )*/
            }
        }
    }

    private fun loadConfig(baseDirectory: File): List<MLConfiguration> {
        val file = Paths.get(baseDirectory.path, "simulation.config").toFile()
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
        trainDataSetIterator: CustomBaseDatasetIterator,
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
        fullTestDataSetIterator: CustomBaseDatasetIterator,

        // Training the network
        trainDataSetIterator: DataSetIterator,
        trainConfiguration: TrainConfiguration,
        modelPoisoningConfiguration: ModelPoisoningConfiguration,

        // Integrating and distributing information to peers
        testDataSetIterator: CustomBaseDatasetIterator,
        countPerPeer: Map<Int, Int>,
    ) {
        /*evaluationProcessor.iteration = 0
        execEvaluationProcessor(
            evaluationProcessor,
            fullTestDataSetIterator,
            network,
            EvaluationProcessor.EvaluationData(
                "before", "", 0, network.iterationCount, 0
            ),
            simulationIndex,
            logging
        )*/
        var iterationsToEvaluation = 0
        while (true) {
            if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                iterationsToEvaluation = 0
            }

            // Train
            var endEpoch = false
            try {
                repeat(3) {
                    network.fit(trainDataSetIterator.next())
                }
            } catch (e: NoSuchElementException) {
                endEpoch = true
            }
            iterationsToEvaluation += trainDataSetIterator.batch()
            if (/*iterationsToEvaluation >= iterationsBeforeEvaluation*/true) {
                // Test
                val end = System.currentTimeMillis()
                if (logging) logger.debug { "Evaluating network " }
                evaluationProcessor.iteration = 0
                execEvaluationProcessor(
                    evaluationProcessor,
                    fullTestDataSetIterator,
                    network,
                    EvaluationProcessor.EvaluationData(
                        "before", "", 0, network.iterationCount, 0
                    ),
                    simulationIndex,
                    logging
                )
            }
        }
        /*

        if (logging) {
            network.setListeners(ScoreIterationListener(printScoreIterations))
        }
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
                if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                    iterationsToEvaluation = 0
                }

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
                        val end = System.currentTimeMillis()
                        if (logging) logger.debug { "Evaluating network " }
                        evaluationProcessor.iteration = iterations
                        execEvaluationProcessor(
                            evaluationProcessor,
                            fullTestDataSetIterator,
                            network,
                            EvaluationProcessor.EvaluationData(
                                "before", "", end - start, network.iterationCount, epoch
                            ),
                            simulationIndex,
                            logging
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
                    newOtherModels.putAll(attackVectors)
                    val numPeers = newOtherModels.size + 1
                    val averageParams: INDArray
                    if (numPeers == 1) {
                        if (logging) logger.debug { "No received params => skipping integration evaluation" }
                        averageParams = newParams
                        network.setParameters(averageParams)
                    } else {
                        if (logging) logger.debug { "Params received => executing aggregation rule" }

                        val start2 = System.currentTimeMillis()
    //                        logger.debug {
    //                            "Integrating newOtherModels: ${newOtherModels[0].second.getDouble(0)}, ${newOtherModels[0].second.getDouble(1)}, ${
    //                                newOtherModels[0].second.getDouble(
    //                                    2
    //                                )
    //                            }, ${newOtherModels[0].second.getDouble(3)}"
    //                        }
                        averageParams = gar.integrateParameters(
                            network,
                            oldParams,
                            gradient,
                            newOtherModels,
                            recentOtherModels,
                            testDataSetIterator,
                            countPerPeer,
                            logging
                        )
                        recentOtherModels.addAll(newOtherModels.toList())
                        while (recentOtherModels.size > NUM_RECENT_OTHER_MODELS) {
                            recentOtherModels.removeFirst()
                        }
                        newOtherModels.clear()
                        network.setParameters(averageParams)

                        if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                            val end2 = System.currentTimeMillis()
                            execEvaluationProcessor(
                                evaluationProcessor,
                                fullTestDataSetIterator,
                                network,
                                EvaluationProcessor.EvaluationData(
                                    "after", numPeers.toString(), end2 - start2, iterations, epoch
                                ),
                                simulationIndex,
                                logging
                            )
                        }
                    }

                    // Send new parameters to other peers
                    if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                        if (logging) logger.debug {
                            "Sending model to peers: ${averageParams.getDouble(0)}, ${averageParams.getDouble(1)}, ${
                                averageParams.getDouble(
                                    2
                                )
                            }, ${averageParams.getDouble(3)}"
                        }
                        testDataSetIterator.reset()
                        val sample = testDataSetIterator.next(500)
                        network.setParameters(averageParams)
                        if (logging) logger.debug { "loss => ${network.score(sample)}" }
                        val message = craftMessage(averageParams, trainConfiguration.behavior, random)
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
                }
                oldParams = network.params().dup()
                if (endEpoch) {
                    break
                }
            }
        }
        logger.debug { "Done training the network" }
        evaluationProcessor.done()*/
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
        evaluationData: EvaluationProcessor.EvaluationData,
        simulationIndex: Int,
        logging: Boolean,
    ) {
        testDataSetIterator.reset()
        evaluationProcessor.extraElements = mapOf(
            Pair("before or after averaging", evaluationData.beforeAfterAveraging),
            Pair("#peers included in current batch", evaluationData.numPeers)
        )
        evaluationProcessor.elapsedTime = evaluationData.elapsedTime
        val evaluationListener = CustomEvaluativeListener(testDataSetIterator, 999999)
        evaluationListener.callback = evaluationProcessor
        evaluationListener.invokeListener(
            network,
            simulationIndex,
            logging
        )
    }
}
