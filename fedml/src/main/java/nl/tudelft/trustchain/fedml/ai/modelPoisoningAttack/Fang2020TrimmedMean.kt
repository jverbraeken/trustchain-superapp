package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.NumAttackers
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.nd4j.linalg.api.ndarray.INDArray
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

    private fun toFloatArray(first: INDArray): FloatArray {
        val data = first.data()
        val length = data.length().toInt()
        val indexer = data.indexer() as FloatRawIndexer
        val array = FloatArray(length)
        for (i in 0 until length) {
            array[i] = indexer.getRaw(i.toLong())
        }
        return array
    }

    private fun generateAttackVector(
        numAttackers: Int,
        modelsAsArrays: Array<FloatArray>,
        gradient: INDArray,
        random: Random,
        b: Int
    ): Array<INDArray> {
        val newMatrices = Array(numAttackers) { Array(1) { FloatArray(modelsAsArrays[0].size) }}
        val data = gradient.data()
        for (i in modelsAsArrays[0].indices) {
            val elements = FloatArray(modelsAsArrays.size)
            modelsAsArrays.forEachIndexed { j, modelsAsArray -> elements[j] = modelsAsArray[i] }
            val v = data.getFloat(i.toLong())
            if (v < 0) {
                /**
                 * Writing minOrNull and maxOrNull functions out completely as micro-optimization
                 * because this is extremely computationally expensive
                 */
                var max = elements[0]
                for (j in 1..elements.lastIndex) {
                    max = if (max >= elements[j]) max else elements[j]
                }
                if (max > 0) {
                    val size = if (max > 10) 0f else (b * max + EPSILON) - (max)
                    newMatrices.forEach { it[0][i] = max + random.nextFloat() * size }
                }
                else {
                    val size = (max / b) - (max - EPSILON)
                    newMatrices.forEach { it[0][i] = max - EPSILON + random.nextFloat() * size }
                }
            } else {
                var min = elements[0]
                for (j in 1..elements.lastIndex) {
                    val e = elements[j]
                    min = if (min < e) min else e
                }
                if (min.isNaN()) {
                    min = 0.0f
                    logger.error { "Found NaN!!!!!!!!!!" }
                }
                if (min > 0) {
                    val size = (min + EPSILON) - (min / b)
                    newMatrices.forEach { it[0][i] = min / b + random.nextFloat() * size }
                }
                else {
                    min = if (min < -10) -10f else min
                    val size = (min) - (b * min - EPSILON)
                    newMatrices.forEach { it[0][i] = b * min - EPSILON + random.nextFloat() * size }
                }
            }
        }
        logger.debug { "${newMatrices.map { NDArray(it).minNumber() }}; ${newMatrices.map { NDArray(
            it
        ).maxNumber() }}" }
        return newMatrices.map { NDArray(it) }.toTypedArray()
    }
}
