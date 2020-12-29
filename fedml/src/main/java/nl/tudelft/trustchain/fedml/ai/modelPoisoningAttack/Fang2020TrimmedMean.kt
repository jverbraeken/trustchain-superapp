package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.NumAttackers
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.shape.Shape
import org.nd4j.linalg.cpu.nativecpu.NDArray
import kotlin.random.Random

private val logger = KotlinLogging.logger("Fang2020TrimmedMean")

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
        val modelsAsArrays = models.map { it.toFloatVector() }.toTypedArray()
        val result = generateAttackVector(numAttackers.num, modelsAsArrays, gradient, random, b)
        return transformToResult(result)
    }

    private fun generateAttackVector(
        numAttackers: Int,
        modelsAsArrays: Array<FloatArray>,
        gradient: INDArray,
        random: Random,
        b: Int
    ): Array<INDArray> {
        val newMatrices = Array(numAttackers) { Array(1) { FloatArray(modelsAsArrays[0].size) }}
        for (i in modelsAsArrays[0].indices) {
            val elements = FloatArray(modelsAsArrays.size)
            modelsAsArrays.forEachIndexed { j, modelsAsArray -> elements[j] = modelsAsArray[i] }
            val offset: Long = Shape.getOffset(gradient.shapeInfoJava(), 0, i)
            val v = gradient.data().getDouble(offset)
            if (v < 0) {
                /**
                 * Writing minOrNull and maxOrNull functions out completely as micro-optimization
                 * because this is extremely computationally expensive
                 */
                var max = elements[0]
                for (j in 1..elements.lastIndex) {
                    max = maxOf(max, elements[j])
                }
                val maxd = max.toDouble()
                if (maxd > 0) newMatrices.forEach { it[0][i] = random.nextDouble(maxd, b * maxd).toFloat() }
                else newMatrices.forEach { it[0][i] = random.nextDouble(maxd, maxd / b).toFloat() }
            } else {
                var min = elements[0]
                for (j in 1..elements.lastIndex) {
                    val e = elements[j]
                    min = minOf(min, e)
                }
                val mind = min.toDouble()
                if (mind > 0) newMatrices.forEach { it[0][i] = random.nextDouble(mind / b, mind).toFloat() }
                else newMatrices.forEach { it[0][i] = random.nextDouble(b * mind, mind).toFloat() }
            }
        }
        return newMatrices.map { NDArray(it) }.toTypedArray()
    }
}
