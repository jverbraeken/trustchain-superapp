package nl.tudelft.trustchain.fedml.ai.dataset

abstract class DatasetManager {
    protected fun findMatchingImageIndices(label: Int, tmpLabelsArr: IntArray): IntArray {
        return (tmpLabelsArr.indices).filter { j: Int -> label == tmpLabelsArr[j] }.toIntArray()
    }

    fun calculateTotalExamples(iteratorDistribution: IntArray, maxTestSamples: Int, labelsArray: IntArray): Int {
        return iteratorDistribution.indices.map { i: Int ->
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
