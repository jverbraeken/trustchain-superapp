package nl.tudelft.trustchain.fedml.ai.dataset

import java.util.*

const val NUM_FULL_TEST_SAMPLES = 100

abstract class DatasetManager {
    protected fun findMatchingImageIndices(label: Int, tmpLabelsArr: Array<Int>): Array<Int> {
        return (tmpLabelsArr.indices).filter { j: Int -> label == tmpLabelsArr[j] }.toTypedArray()
    }

    protected fun shuffle(matchingImageIndices: Array<Int>, seed: Long): Array<Int> {
        val tmp = matchingImageIndices.clone().toMutableList()
        tmp.shuffle(Random(seed))
        return tmp.toTypedArray()
    }

    fun calculateTotalExamples(iteratorDistribution: List<Int>?, maxTestSamples: Int, labelsArray: Array<Int>): Int {
        return iteratorDistribution?.indices?.map { i: Int ->
            minOf(
                iteratorDistribution[i],
                maxTestSamples,
                labelsArray
                    .filter { j: Int -> j == i }
                    .size
            )
        }?.sum()
            ?: labelsArray.distinct().size * NUM_FULL_TEST_SAMPLES
    }
}
