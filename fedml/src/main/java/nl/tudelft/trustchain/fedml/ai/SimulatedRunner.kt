package nl.tudelft.trustchain.fedml.ai

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread
import kotlin.random.Random

private val logger = KotlinLogging.logger("SimulatedRunner")

class SimulatedRunner : Runner() {
    private val paramBuffers: MutableList<CopyOnWriteArrayList<INDArray>> = ArrayList()
    private val randoms: MutableList<Random> = ArrayList()

    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration
    ) {
        val numThreads = 2
        for (i in 0 until numThreads) {
            paramBuffers.add(CopyOnWriteArrayList())
            randoms.add(Random(i))
            thread {
                val trainDataSetIterator = getTrainDatasetIterator(
                    baseDirectory,
                    mlConfiguration.dataset,
                    mlConfiguration.datasetIteratorConfiguration,
                    mlConfiguration.trainConfiguration.behavior,
                    i
                )
                val testDataSetIterator = getTestDatasetIterator(
                    baseDirectory,
                    mlConfiguration.dataset,
                    mlConfiguration.datasetIteratorConfiguration,
                    mlConfiguration.trainConfiguration.behavior,
                    i
                )
                val evaluationProcessor = EvaluationProcessor(
                    baseDirectory,
                    "simulated",
                    mlConfiguration,
                    i,
                    listOf(
                        "before or after averaging",
                        "#peers included in current batch"
                    ),
                    "-simulation-$i"
                )
                val network = generateNetwork(
                    mlConfiguration.dataset,
                    mlConfiguration.nnConfiguration,
                    i
                )
                network.setListeners(ScoreIterationListener(printScoreIterations))

                trainTestSendNetwork(
                    i,
                    network,
                    evaluationProcessor,
                    trainDataSetIterator,
                    testDataSetIterator,
                    mlConfiguration.trainConfiguration,
                    mlConfiguration.modelPoisoningConfiguration
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
        modelPoisoningConfiguration: ModelPoisoningConfiguration
    ) {
        val paramBuffer = paramBuffers[simulationIndex]
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

                    // Integrate parameters of other peers
                    network.setParams(oldParams)
                    val attack = modelPoisoningConfiguration.attack
                    val attackVectors = attack.obj.generateAttack(
                        modelPoisoningConfiguration.numAttackers,
                        oldParams,
                        gradient,
                        paramBuffer,
                        random
                    )
                    paramBuffer.addAll(attackVectors)
                    val numPeers = paramBuffer.size + 1
                    val averageParams: INDArray
                    if (numPeers == 1) {
                        logger.debug { "No received params => skipping integration evaluation" }
                        averageParams = newParams
                    } else {
                        logger.debug { "Params received => executing aggregation rule" }

                        val start2 = System.currentTimeMillis()
                        averageParams = gar.integrateParameters(oldParams, gradient, paramBuffer, network, testDataSetIterator)
                        paramBuffer.clear()
                        network.setParameters(averageParams)
                        oldParams = averageParams.dup()
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

                    // Send new parameters to other peers
                    val message = craftMessage(averageParams, trainConfiguration.behavior, random)
                    when (trainConfiguration.communicationPattern) {
                        CommunicationPatterns.ALL -> paramBuffers.filterIndexed { index, _ -> index != simulationIndex }
                            .forEach { it.add(message) }
                        CommunicationPatterns.RANDOM -> paramBuffers.filterIndexed { index, _ -> index != simulationIndex }
                            .random().add(message)
                        CommunicationPatterns.RR -> throw IllegalArgumentException("Not implemented yet")
                        CommunicationPatterns.RING -> throw IllegalArgumentException("Not implemented yet")
                    }
                }
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
}
