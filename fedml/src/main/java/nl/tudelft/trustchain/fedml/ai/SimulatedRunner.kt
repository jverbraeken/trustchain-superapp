package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File

class SimulatedRunner : Runner() {
    private val aggregationRule: AggregationRule = SimpleAggregator()

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
            var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "simulated",
                mlConfiguration.dataset.text,
                mlConfiguration.optimizer.text,
                mlConfiguration.learningRate.text,
                mlConfiguration.momentum?.text ?: "null",
                mlConfiguration.l2.text,
                mlConfiguration.batchSize.text,
                mlConfiguration.iteratorDistribution.text,
                mlConfiguration.maxTestSamples.text,
                seed,
                listOf("before or after averaging")
            )
            evaluationListener.callback = evaluationProcessor

            val networks = arrayOf(
                generateNetwork(
                    mlConfiguration.dataset,
                    mlConfiguration.optimizer,
                    mlConfiguration.learningRate,
                    mlConfiguration.momentum,
                    mlConfiguration.l2,
                    seed
                ),
                generateNetwork(
                    mlConfiguration.dataset,
                    mlConfiguration.optimizer,
                    mlConfiguration.learningRate,
                    mlConfiguration.momentum,
                    mlConfiguration.l2,
                    seed
                )
            )
            networks.forEach { it.setListeners(ScoreIterationListener(printScoreIterations)) }

            var epoch = 0
            var iterations = 0
            for (i in 0 until mlConfiguration.epoch.value) {
                epoch++
                evaluationProcessor.epoch = epoch
                val start = System.currentTimeMillis()
                while (true) {
                    var endEpoch = false
                    for (net in networks) {
                        for (j in 0 until mlConfiguration.batchSize.value) {
                            try {
                                net.fit(trainDataSetIterator.next())
                            } catch (e: NoSuchElementException) {
                                endEpoch = true
                            }
                        }
                    }
                    iterations += mlConfiguration.batchSize.value
                    val end = System.currentTimeMillis()

                    evaluationProcessor.iteration = iterations
                    evaluationProcessor.extraElements =
                        mapOf(Pair("before or after averaging", "before"))
                    evaluationProcessor.elapsedTime = end - start
                    evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                    evaluationListener.callback = evaluationProcessor
                    evaluationListener.iterationDone(networks[0], networks[0].iterationCount, epoch)

                    val params: MutableList<Pair<INDArray, Int>> = ArrayList(networks.size)
                    networks.forEach { params.add(Pair(it.params().dup(), mlConfiguration.batchSize.value)) }
                    val averageParams = aggregationRule.integrateParameters(
                        params[0],
                        params.subList(1, params.size),
                        networks[0],
                        testDataSetIterator
                    )
                    networks.forEach { it.setParams(averageParams.first) }

                    evaluationProcessor.extraElements =
                        mapOf(Pair("before or after averaging", "after"))
                    evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                    evaluationListener.callback = evaluationProcessor
                    evaluationListener.iterationDone(networks[0], iterations, epoch)
                    if (endEpoch)
                        break
                }
            }
            evaluationProcessor.done()
        }
    }
}
