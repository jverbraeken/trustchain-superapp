package nl.tudelft.trustchain.fedml.ai.dataset

import mu.KotlinLogging

private val logger = KotlinLogging.logger("CustomMnistManager")

abstract class DatasetManager {
    fun calculateTotalExamples(labelIndexMapping: Array<IntArray>, iteratorDistribution: IntArray, maxTestSamples: Int): Int {
        return iteratorDistribution.indices.map { i ->
            minOf(labelIndexMapping[i].size, iteratorDistribution[i], maxTestSamples)
        }.sum()
    }
}
