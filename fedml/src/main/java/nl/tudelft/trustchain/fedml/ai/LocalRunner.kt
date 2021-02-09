package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import java.io.File


private val logger = KotlinLogging.logger("LocalRunner")

class LocalRunner : Runner() {
    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration,
    ) {
        val iterationsBeforeEvaluation = 5

        scope.launch {
            val trainDataSetIterator = mlConfiguration.dataset.inst(
                mlConfiguration.datasetIteratorConfiguration,
                seed.toLong(),
                CustomDataSetType.TRAIN,
                baseDirectory,
                Behaviors.BENIGN
            )
            val testDataSetIterator = mlConfiguration.dataset.inst(
                mlConfiguration.datasetIteratorConfiguration,
                seed.toLong() + 1,
                CustomDataSetType.FULL_TEST,
                baseDirectory,
                Behaviors.BENIGN
            )
            val network = generateNetwork(
                mlConfiguration.dataset.architecture,
                mlConfiguration.nnConfiguration,
                seed
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
            evaluationProcessor.done()
        }
    }
}
