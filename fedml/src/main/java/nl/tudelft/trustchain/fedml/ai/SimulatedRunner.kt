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
        numEpochs: Epochs,
        dataset: Datasets,
        optimizer: Optimizers,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations,
        batchSize: BatchSizes,
        iteratorDistribution: IteratorDistributions,
        maxTestSamples: MaxTestSamples,
        seed: Int
    ) {
        scope.launch {
            val trainDataSetIterator = getTrainDatasetIterator(
                baseDirectory,
                dataset,
                batchSize,
                iteratorDistribution,
                seed
            )
            val testDataSetIterator = getTestDatasetIterator(
                baseDirectory,
                dataset,
                batchSize,
                iteratorDistribution,
                seed,
                maxTestSamples
            )
            var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "simulated",
                dataset.text,
                optimizer.text,
                learningRate.text,
                momentum?.text ?: "null",
                l2.text,
                batchSize.text,
                listOf("before or after averaging")
            )
            evaluationListener.callback = evaluationProcessor

            val networks = arrayOf(
                generateNetwork(dataset, optimizer, learningRate, momentum, l2, seed),
                generateNetwork(dataset, optimizer, learningRate, momentum, l2, seed)
            )
            networks.forEach { it.setListeners(ScoreIterationListener(printScoreIterations)) }

            var epoch = 0
            var iterations = 0
            for (i in 0 until numEpochs.value) {
                epoch++
                evaluationProcessor.epoch = epoch
                val start = System.currentTimeMillis()
                while (true) {
                    var endEpoch = false
                    for (net in networks) {
                        for (j in 0 until batchSize.value) {
                            try {
                                net.fit(trainDataSetIterator.next())
                            } catch (e: NoSuchElementException) {
                                endEpoch = true
                            }
                        }
                    }
                    iterations += batchSize.value
                    val end = System.currentTimeMillis()

                    evaluationProcessor.iteration = iterations
                    evaluationProcessor.extraElements =
                        mapOf(Pair("before or after averaging", "before"))
                    evaluationProcessor.elapsedTime = end - start
                    evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                    evaluationListener.callback = evaluationProcessor
                    evaluationListener.iterationDone(networks[0], networks[0].iterationCount, epoch)

                    val params: MutableList<Pair<INDArray, Int>> = ArrayList(networks.size)
                    networks.forEach { params.add(Pair(it.params(), batchSize.value)) }
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
