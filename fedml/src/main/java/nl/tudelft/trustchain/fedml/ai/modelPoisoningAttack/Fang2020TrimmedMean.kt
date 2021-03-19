package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.NumAttackers
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.shape.Shape
import org.nd4j.linalg.cpu.nativecpu.NDArray
import kotlin.random.Random

private val logger = KotlinLogging.logger("Fang2020TrimmedMean")
private const val EPSILON = 1e-5f

/**
 * IMPORTANT: b refers in this code and the original paper NOT to the amount of attackers, but to the attacker's range
 */
class Fang2020TrimmedMean(private val b: Int) : ModelPoisoningAttack() {
    override fun generateAttack(
        numAttackers: NumAttackers,
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: Map<Int, INDArray>,
        random: Random
    ): Map<Int, INDArray> {
        logger.debug { formatName("Fang 2020 Trimmed Mean") }
        val models = arrayOf<INDArray>(oldModel.sub(gradient), *(otherModels.values.toTypedArray()))
        logger.debug { "Found ${models.size} models in total" }
        if (models.size < 4) {
            logger.debug { "Too few models => no attack vectors generated" }
            return mapOf()
        }
        val modelsAsArrays = models.map { toFloatArray(it) }.toTypedArray()
        val result = generateAttackVector(numAttackers.num, modelsAsArrays, gradient, random, b)
        return transformToResult(result)
    }

    private fun toFloatArray(first: INDArray): Array<FloatArray> {
        val data = first.data()
        val array = Array(first.rows()) { FloatArray(first.columns()) }
        val indexer = data.indexer() as FloatRawIndexer
        val shape = first.shapeInfoJava()
        for (i in 0 until first.rows()) {
            for (j in 0 until first.columns()) {
                val offset = Shape.getOffset(shape, i, j)
                array[i][j] = indexer.getRaw(offset)
            }
        }
        return array
    }

    private fun generateAttackVector(
        numAttackers: Int,
        modelsAsArrays: Array<Array<FloatArray>>,
        gradient: INDArray,
        random: Random,
        b: Int
    ): Array<INDArray> {
        val newMatrices = Array(numAttackers) { Array(modelsAsArrays[0].size) { FloatArray(modelsAsArrays[0][0].size) }}
        val data = gradient.data()
        val indexer = data.indexer() as FloatRawIndexer
        val shape = gradient.shapeInfoJava()
        for (i in modelsAsArrays[0].indices) {
            for (j in modelsAsArrays[0][0].indices) {
                val offset = Shape.getOffset(shape, i, j)
                val v = indexer.getRaw(offset)
                if (v < 0) {
                    /**
                     * Writing minOrNull and maxOrNull functions out completely as micro-optimization
                     * because this is extremely computationally expensive
                     */
                    var max = modelsAsArrays[0][i][j]
                    for (k in 1..modelsAsArrays.lastIndex) {
                        val n = modelsAsArrays[k][i][j]
                        max = if (max >= n) max else n
                    }
                    if (max > 0) {
                        val size = if (max > 10) 0f else (b * max + EPSILON) - (max)
                        newMatrices.forEach { it[i][j] = max + random.nextFloat() * size }
                    } else {
                        val size = (max / b) - (max - EPSILON)
                        newMatrices.forEach { it[i][j] = max - EPSILON + random.nextFloat() * size }
                    }
                } else {
                    var min = modelsAsArrays[0][i][j]
                    for (k in 1..modelsAsArrays.lastIndex) {
                        val n = modelsAsArrays[k][i][j]
                        min = if (min < n) min else n
                    }
                    if (min.isNaN()) {
                        min = 0.0f
                        logger.error { "Found NaN!!!!!!!!!!" }
                    }
                    if (min > 0) {
                        val size = (min + EPSILON) - (min / b)
                        newMatrices.forEach { it[i][j] = min / b + random.nextFloat() * size }
                    } else {
                        min = if (min < -10) -10f else min
                        val size = (min) - (b * min - EPSILON)
                        newMatrices.forEach { it[i][j] = b * min - EPSILON + random.nextFloat() * size }
                    }
                }
            }
        }
        logger.debug { "${newMatrices.map { NDArray(it).minNumber() }}; ${newMatrices.map { NDArray(
            it
        ).maxNumber() }}" }
        return newMatrices.map { NDArray(it) }.toTypedArray()
    }
}
