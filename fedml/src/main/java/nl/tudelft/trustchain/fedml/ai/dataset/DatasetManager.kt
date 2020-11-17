package nl.tudelft.trustchain.fedml.ai.dataset

import java.util.*
import java.util.stream.Collectors
import java.util.stream.IntStream

abstract class DatasetManager {
    protected fun findMatchingImageIndices(label: Int, tmpLabelsArr: IntArray): IntArray {
        return IntStream.range(0, tmpLabelsArr.size).filter { j: Int -> label == tmpLabelsArr[j] }.toArray()
    }

    protected fun shuffle(matchingImageIndices: IntArray, seed: Long): IntArray {
        val tmp = Arrays.stream(matchingImageIndices.clone()).boxed().collect(Collectors.toList())
        Collections.shuffle(tmp, Random(seed))
        return tmp.stream().mapToInt { j: Int? -> j!! }.toArray()
    }

    protected fun min(a: Int, b: Int, c: Int): Int {
        return Math.min(Math.min(a, b), c)
    }

    fun calculateTotalExamples(iteratorDistribution: List<Int>, maxTestSamples: Int, labelsArray: IntArray?): Int {
        return IntStream.range(0, iteratorDistribution.size).map { i: Int ->
            min(
                iteratorDistribution[i],
                maxTestSamples, Arrays.stream(labelsArray).filter { j: Int -> j == i }.count()
                    .toInt()
            )
        }.sum()
    }
}
