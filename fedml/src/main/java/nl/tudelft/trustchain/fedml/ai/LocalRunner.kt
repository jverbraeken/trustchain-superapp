package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import java.io.File

class LocalRunner : Runner() {
    override fun run(
        baseDirectory: File,
        numEpochs: Epochs,
        dataset: Datasets,
        optimizer: Optimizers,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations,
        batchSize: BatchSizes
    ) {
        scope.launch {
            val trainDataSetIterator = getTrainDatasetIterator(baseDirectory, dataset, batchSize)
            val testDataSetIterator = getTestDatasetIterator(baseDirectory, dataset, batchSize)
            val network = generateNetwork(dataset, optimizer, learningRate, momentum, l2)
            var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "local",
                dataset.identifier,
                optimizer.identifier,
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
            for (i in 0 until numEpochs.value) {
                epoch++
                evaluationProcessor.epoch = epoch
                val start = System.currentTimeMillis()
                loop@ while (true) {
                    for (j in 0 until batchSize.value) {
                        try {
                            network.fit(trainDataSetIterator.next())
                        } catch (e: NoSuchElementException) {
                            break@loop
                        }
                    }
                    iterations += batchSize.value
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
