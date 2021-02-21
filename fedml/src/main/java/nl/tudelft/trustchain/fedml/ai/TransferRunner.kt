package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.BatchSizes
import nl.tudelft.trustchain.fedml.Behaviors
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import java.io.File
import kotlin.math.min


private val logger = KotlinLogging.logger("LocalRunner")
private const val ITERATIONS_BEFORE_EVALUATION = 1000

class TransferRunner : Runner() {
    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration,
    ) {
        scope.launch {
            val trainDataSetIterator = mlConfiguration.dataset.inst(
                DatasetIteratorConfiguration(
                    mlConfiguration.datasetIteratorConfiguration.batchSize,
                    mlConfiguration.datasetIteratorConfiguration.distribution,
                    mlConfiguration.datasetIteratorConfiguration.maxTestSamples
                ),
                seed.toLong(),
                CustomDataSetType.TRAIN,
                baseDirectory,
                Behaviors.BENIGN,
                true,
            )
            logger.debug { "Loaded trainDataSetIterator" }
            val testDataSetIterator = mlConfiguration.dataset.inst(
                DatasetIteratorConfiguration(
                    BatchSizes.BATCH_200,
                    mlConfiguration.datasetIteratorConfiguration.distribution.map { min(10, it) },
                    mlConfiguration.datasetIteratorConfiguration.maxTestSamples
                ),
                (seed + 1).toLong(),
                CustomDataSetType.TEST,
                baseDirectory,
                Behaviors.BENIGN,
                true,
            )
            logger.debug { "Loaded testDataSetIterator" }
            val network = generateNetwork(
                mlConfiguration.dataset.architectureTransfer,
                mlConfiguration.nnConfiguration,
                seed
            )
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "local",
                ArrayList()
            )
            evaluationProcessor.newSimulation("transfer", listOf(mlConfiguration), false)
            network.setListeners(
                ScoreIterationListener(printScoreIterations)
            )

            var epoch = 0
            var iterations = 0
            var iterationsToEvaluation = 0
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

                    if (endEpoch) {
                        ModelSerializer.writeModel(network, File(baseDirectory, "transfer-${mlConfiguration.dataset.id}"), false)
                        break
                    }

                    if (iterationsToEvaluation % ITERATIONS_BEFORE_EVALUATION == 0) {
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
                }

                if (epoch >= 2) {
                    break@epochLoop
                }
            }
            logger.debug { "Done training the network" }
            evaluationProcessor.done()
        }
    }
}
