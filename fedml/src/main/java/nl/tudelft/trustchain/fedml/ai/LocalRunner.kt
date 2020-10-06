package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import java.io.File

class LocalRunner : Runner() {
    override fun run(
        baseDirectory: File,
        dataset: Datasets,
        updater: Updaters,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations,
        batchSize: BatchSizes
    ) {
        val trainDataSetIterator = getTrainDatasetIterator(baseDirectory, dataset, batchSize)
        val testDataSetIterator = getTestDatasetIterator(baseDirectory, dataset, batchSize)
        val network = generateNetwork(dataset, updater, learningRate, momentum, l2)
        var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
        val evaluationProcessor = EvaluationProcessor(
            baseDirectory,
            "local",
            dataset.identifier,
            updater.identifier,
            learningRate.identifier,
            momentum?.identifier ?: "null",
            l2.identifier,
            batchSize.identifier,
            ArrayList()
        )
        evaluationListener.callback = evaluationProcessor
        network.setListeners(
            ScoreIterationListener(printScoreIterations),
            evaluationListener
        )

        var epoch = 0
        var iterations = 0
        val numEpochs = 10
        for (i in 0 until numEpochs) {
            epoch++
            evaluationProcessor.epoch = epoch
            loop@while (true) {
                for (j in 0 until batchSize.value) {
                    try {
                        network.fit(trainDataSetIterator.next())
                    } catch (e: NoSuchElementException) {
                        break@loop
                    }
                }
                iterations += batchSize.value
                evaluationProcessor.iteration = iterations
                evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                evaluationListener.callback = evaluationProcessor
                evaluationListener.iterationDone(network, iterations, epoch)
            }
        }
        evaluationProcessor.done()
    }
}
