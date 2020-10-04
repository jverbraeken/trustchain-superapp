package nl.tudelft.trustchain.fedml.ai

import org.nd4j.linalg.api.ndarray.INDArray

class SimulatedRunner : Runner() {
    override fun run(
        dataset: Datasets,
        updater: Updaters,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations,
        batchSize: BatchSizes
    ) {
        val networks = arrayOf(
            generateNetwork(dataset, updater, learningRate, momentum, l2),
            generateNetwork(dataset, updater, learningRate, momentum, l2)
        )
        while (true) {
            for (net in networks) {
                for (i in 0 until batchSize.value) {
                    net.fit(getTrainDatasetIterator(dataset, batchSize).next())
                }
            }
            val params : MutableList<Pair<INDArray, Int>> = ArrayList(networks.size)
            networks.forEach { params.add(Pair(it.params(), batchSize.value)) }
            val averageParams = calculateWeightedAverageParams(params)
            networks.forEach { it.setParams(averageParams.first)}
        }
    }
}
