package nl.tudelft.trustchain.fedml.ai.dataset

import java.util.*
import java.util.stream.Collectors
import java.util.stream.IntStream

abstract class DatasetManager {
    protected fun findMatchingImageIndices(label: Int, tmpLabelsArr: IntArray): IntArray {
        return IntStream.range(0, tmpLabelsArr.size).filter { j: Int -> label == tmpLabelsArr[j] }.toArray()
    }

    protected fun shuffle(matchingImageIndices: IntArray, seed: Long): Array<Int> {
        val tmp = matchingImageIndices.clone().toMutableList()
        tmp.shuffle(Random(seed))
        return tmp.toTypedArray()
    }

    fun calculateTotalExamples(iteratorDistribution: List<Int>, maxTestSamples: Int, labelsArray: IntArray): Int {
        return IntStream.range(0, iteratorDistribution.size).map { i: Int ->
            minOf(
                iteratorDistribution[i],
                maxTestSamples,
                labelsArray
                    .filter { j: Int -> j == i }
                    .size
            )
        }.sum()
    }
}
