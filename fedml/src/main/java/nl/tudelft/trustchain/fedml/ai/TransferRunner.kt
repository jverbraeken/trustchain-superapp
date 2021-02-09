package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import java.io.File


private val logger = KotlinLogging.logger("LocalRunner")

class TransferRunner : Runner() {
    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration,
    ) {
        val iterationsBeforeEvaluation = 1000

        scope.launch {
            val trainDataSetIterator = mlConfiguration.dataset.instTransfer(
                mlConfiguration.datasetIteratorConfiguration.batchSize.value,
                true
            )
            val testDataSetIterator = mlConfiguration.dataset.instTransfer(
                mlConfiguration.datasetIteratorConfiguration.batchSize.value,
                false
            )
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
                ModelSerializer.writeModel(network, File(baseDirectory, "transferMnistTemp"), false)
                while (true) {
                    var endEpoch = false
                    try {
                        network.fit(trainDataSetIterator.next())
                    } catch (e: NoSuchElementException) {
                        endEpoch = true
                    }
                    iterations += 1
                    iterationsToEvaluation += 1
                    if (epoch >= 5) {
                        break@epochLoop
                    }
                    if (endEpoch) {
                        break
                    }
                }
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

                ModelSerializer.writeModel(network, File(baseDirectory, "transferMnist"), false)
                break
            }
            logger.debug { "Done training the network" }
            evaluationProcessor.done()
        }
    }
}
