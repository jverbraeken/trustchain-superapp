package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.Fang2020Krum
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.Fang2020TrimmedMean
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.opencv.dnn.Net
import java.io.File
import kotlin.random.Random


private val logger = KotlinLogging.logger("LocalRunner")

class LocalRunner : Runner() {
    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration,
    ) {
        val iterationsBeforeEvaluation = 5


        val evaluationProcessor = EvaluationProcessor(
            baseDirectory,
            "local",
            ArrayList()
        )

        val nodes = (0 until 10).map {
            Node(
                0,
                MLConfiguration(
                    Datasets.MNIST,
                    DatasetIteratorConfiguration(
                        Datasets.MNIST.defaultBatchSize,
                        listOf(2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                        MaxTestSamples.NUM_200
                    ),
                    NNConfiguration(
                        Datasets.MNIST.defaultOptimizer,
                        Datasets.MNIST.defaultLearningRate,
                        Datasets.MNIST.defaultMomentum,
                        Datasets.MNIST.defaultL2
                    ),
                    TrainConfiguration(
                        MaxIterations.ITER_300,
                        GARs.NONE,
                        CommunicationPatterns.ALL,
                        Behaviors.BENIGN,
                        Slowdowns.NONE,
                        TransmissionRounds.N0,
                        10,
                        1,
                        true,
                        1.0,
                        0
                    ),
                    ModelPoisoningConfiguration(ModelPoisoningAttacks.NONE, NumAttackers.NUM_0)
                ),
                ::generateNetwork,
                ::getDataSetIterators,
                baseDirectory,
                evaluationProcessor,
                System.currentTimeMillis(),
                ::shareModel
            )
        }
        repeat(10) { iter ->
            nodes.forEach { it.performIteration(0, iter) }
        }

        val nodeLabelFlip = Node(
            0,
            MLConfiguration(
                Datasets.MNIST,
                DatasetIteratorConfiguration(Datasets.MNIST.defaultBatchSize, listOf(2, 2, 2, 2, 2, 2, 2, 2, 2, 2), MaxTestSamples.NUM_200),
                NNConfiguration(Datasets.MNIST.defaultOptimizer, Datasets.MNIST.defaultLearningRate, Datasets.MNIST.defaultMomentum, Datasets.MNIST.defaultL2),
                TrainConfiguration(MaxIterations.ITER_300, GARs.NONE, CommunicationPatterns.ALL, Behaviors.LABEL_FLIP_ALL, Slowdowns.NONE, TransmissionRounds.N0, 10, 1, true, 1.0, 0),
                ModelPoisoningConfiguration(ModelPoisoningAttacks.NONE, NumAttackers.NUM_0)
            ),
            ::generateNetwork,
            ::getDataSetIterators,
            baseDirectory,
            evaluationProcessor,
            System.currentTimeMillis(),
            ::shareModel
        )
        repeat(10) {
            nodeLabelFlip.performIteration(0, it)
        }
        val distanceLabelFlip = nodes[0].newParams.distance2(nodeLabelFlip.newParams)
        logger.error { "distanceLabelFlip: $distanceLabelFlip" }

        val nodeNoise = Node(
            0,
            MLConfiguration(
                Datasets.MNIST,
                DatasetIteratorConfiguration(Datasets.MNIST.defaultBatchSize, listOf(2, 2, 2, 2, 2, 2, 2, 2, 2, 2), MaxTestSamples.NUM_200),
                NNConfiguration(Datasets.MNIST.defaultOptimizer, Datasets.MNIST.defaultLearningRate, Datasets.MNIST.defaultMomentum, Datasets.MNIST.defaultL2),
                TrainConfiguration(MaxIterations.ITER_300, GARs.NONE, CommunicationPatterns.ALL, Behaviors.NOISE, Slowdowns.NONE, TransmissionRounds.N0, 10, 1, true, 1.0, 0),
                ModelPoisoningConfiguration(ModelPoisoningAttacks.NONE, NumAttackers.NUM_0)
            ),
            ::generateNetwork,
            ::getDataSetIterators,
            baseDirectory,
            evaluationProcessor,
            System.currentTimeMillis(),
            ::shareModel
        )
        repeat(10) {
            nodeNoise.performIteration(0, it)
        }
        val distanceNoise = nodes[0].newParams.distance2(nodeNoise.newParams)
        logger.error { "distanceNoise: $distanceNoise" }

        Net().let {
            val newParams = nodes[0].network.outputLayer.paramTable().getValue("W").dup()
            val gradient = nodes[0].oldParams.sub(newParams)
            val attack = Fang2020Krum(4).generateAttack(
                NumAttackers.NUM_1,
                nodes[0].oldParams,
                gradient,
                mapOf(Pair(0, nodes[1].newParams), Pair(0, nodes[2].newParams), Pair(0, nodes[3].newParams)),
                Random(1)
            )
            val distanceKrum = nodes[0].newParams.distance2(attack.toList().first().second)
            logger.error { "distanceKrum: $distanceKrum" }
        }

        Net().let {
            val newParams = nodes[0].network.outputLayer.paramTable().getValue("W").dup()
            val gradient = nodes[0].oldParams.sub(newParams)
            val attack = Fang2020TrimmedMean(4).generateAttack(
                NumAttackers.NUM_1,
                nodes[0].oldParams,
                gradient,
                mapOf(Pair(0, nodes[1].newParams), Pair(0, nodes[2].newParams), Pair(0, nodes[3].newParams)),
                Random(1)
            )
            val distanceTrimmedMean = nodes[0].newParams.distance2(attack.toList().first().second)
            logger.error { "distanceTrimmedMean: $distanceTrimmedMean" }
        }

































































//        scope.launch {
        /*val trainDataSetIterator = mlConfiguration.dataset.inst(
            mlConfiguration.datasetIteratorConfiguration,
            seed.toLong(),
            CustomDataSetType.TRAIN,
            baseDirectory,
            Behaviors.BENIGN,
            false,
        )
        val testDataSetIterator = mlConfiguration.dataset.inst(
            mlConfiguration.datasetIteratorConfiguration,
            seed.toLong() + 1,
            CustomDataSetType.FULL_TEST,
            baseDirectory,
            Behaviors.BENIGN,
            false,
        )
        val network = generateNetwork(
            mlConfiguration.dataset.architecture,
            mlConfiguration.nnConfiguration,
            seed,
            NNConfigurationMode.REGULAR
        )
        val evaluationProcessor = EvaluationProcessor(
            baseDirectory,
            "local",
            ArrayList()
        )
        evaluationProcessor.newSimulation("local run", listOf(mlConfiguration), false)
        network.setListeners(
            ScoreIterationListener(printScoreIterations)
        )

        var epoch = 0
        var iterations = 0
        var iterationsToEvaluation = 0
        val trainConfiguration = mlConfiguration.trainConfiguration
        epochLoop@ while (true) {
            epoch++
            trainDataSetIterator.reset()
            logger.debug { "Starting epoch: $epoch" }
            val start = System.currentTimeMillis()
            while (true) {
                var endEpoch = false
                try {
                    network.fit(trainDataSetIterator.next())
                } catch (e: NoSuchElementException) {
                    endEpoch = true
                }
                iterations += 1
                iterationsToEvaluation += 1
                logger.debug { "Iteration: $iterations" }

                if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                    iterationsToEvaluation = 0
                    val end = System.currentTimeMillis()
                    evaluationProcessor.evaluate(
                        testDataSetIterator,
                        network,
                        mapOf(),
                        end - start,
                        iterations,
                        epoch,
                        0,
                        true
                    )
                }
                if (iterations >= trainConfiguration.maxIteration.value) {
                    break@epochLoop
                }
                if (endEpoch) {
                    break
                }
            }
        }
        logger.debug { "Done training the network" }
        evaluationProcessor.done()*/
//        }
    }

    private fun shareModel(
        params: INDArray,
        trainConfiguration: TrainConfiguration,
        random: Random,
        nodeIndex: Int,
        countPerPeer: Map<Int, Int>
    ) {

    }
}
