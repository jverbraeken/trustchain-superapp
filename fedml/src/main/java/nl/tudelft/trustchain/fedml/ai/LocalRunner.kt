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
                seed.toLong(),
                CustomDataSetType.TEST,
                baseDirectory,
                Behaviors.BENIGN
            )
            val network = generateNetwork(
                mlConfiguration.dataset,
                mlConfiguration.nnConfiguration,
                seed
            )
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "local",
                ArrayList()
            )
            evaluationProcessor.newSimulation("local run", listOf(mlConfiguration))
            network.setListeners(
                ScoreIterationListener(printScoreIterations)
            )

            var epoch = 0
            var iterations = 0
            var iterationsToEvaluation = 0
            val trainConfiguration = mlConfiguration.trainConfiguration
            val datasetIteratorConfiguration = mlConfiguration.datasetIteratorConfiguration
            for (i in 0 until trainConfiguration.numEpochs.value) {
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
                    iterations += datasetIteratorConfiguration.batchSize.value
                    iterationsToEvaluation += datasetIteratorConfiguration.batchSize.value

                    if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                        iterationsToEvaluation = 0
                        val end = System.currentTimeMillis()
                        evaluationProcessor.evaluate(
                            testDataSetIterator,
                            network,
                            mapOf(
                                Pair("before or after averaging", "after"),
                                Pair("#peers included in current batch", "")
                            ),
                            end - start,
                            iterations,
                            epoch,
                            0,
                            true
                        )
                    }
                    if (endEpoch) {
                        break
                    }
                }
            }
            evaluationProcessor.done()
        }
    }
}
