package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import java.io.File

class LocalRunner : Runner() {
    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration
    ) {
        scope.launch {
            val trainDataSetIterator = getTrainDatasetIterator(
                baseDirectory,
                mlConfiguration.dataset,
                mlConfiguration.batchSize,
                mlConfiguration.iteratorDistribution,
                seed
            )
            val testDataSetIterator = getTestDatasetIterator(
                baseDirectory,
                mlConfiguration.dataset,
                mlConfiguration.batchSize,
                mlConfiguration.iteratorDistribution,
                seed,
                mlConfiguration.maxTestSamples
            )
            val network = generateNetwork(
                mlConfiguration.dataset,
                mlConfiguration.optimizer,
                mlConfiguration.learningRate,
                mlConfiguration.momentum,
                mlConfiguration.l2,
                seed
            )
            var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "local",
                mlConfiguration.dataset.text,
                mlConfiguration.optimizer.text,
                mlConfiguration.learningRate.text,
                mlConfiguration.momentum?.text ?: "null",
                mlConfiguration.l2.text,
                mlConfiguration.batchSize.text,
                mlConfiguration.iteratorDistribution.text,
                mlConfiguration.maxTestSamples.text,
                seed,
                ArrayList()
            )
            evaluationListener.callback = evaluationProcessor
            network.setListeners(
                ScoreIterationListener(printScoreIterations),
                evaluationListener
            )

            var epoch = 0
            var iterations = 0
            for (i in 0 until mlConfiguration.epoch.value) {
                epoch++
                evaluationProcessor.epoch = epoch
                val start = System.currentTimeMillis()
                loop@ while (true) {
                    for (j in 0 until mlConfiguration.batchSize.value) {
                        try {
                            network.fit(trainDataSetIterator.next())
                        } catch (e: NoSuchElementException) {
                            break@loop
                        }
                    }
                    iterations += mlConfiguration.batchSize.value
                    val end = System.currentTimeMillis()
                    evaluationProcessor.iteration = iterations
                    evaluationProcessor.elapsedTime = end - start
                    evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                    evaluationListener.callback = evaluationProcessor
                    evaluationListener.iterationDone(network, iterations, epoch)
                }
            }
            evaluationProcessor.done()
        }
    }
}
