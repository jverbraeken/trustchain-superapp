package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.NumAttackers
import org.nd4j.linalg.api.ndarray.INDArray
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
        val models = arrayListOf<INDArray>(oldModel.sub(gradient))
        models.addAll(otherModels.values)
        logger.debug { "Found ${models.size} models in total" }
        val modelsAsArrays = models.map { it.toFloatMatrix()[0] }.toTypedArray()
        val result = Array(numAttackers.num) { generateAttackVector(modelsAsArrays, gradient, random, b) }
        return transformToResult(result)
    }

    private fun generateAttackVector(
        modelsAsArrays: Array<FloatArray>,
        gradient: INDArray,
        random: Random,
        b: Int
    ): INDArray {
        val newMatrix = Array(1) { FloatArray(modelsAsArrays[0].size) }
        for (i in modelsAsArrays[0].indices) {
            val elements = FloatArray(modelsAsArrays.size)
            modelsAsArrays.forEachIndexed { j, modelsAsArray -> elements[j] = modelsAsArray[i] }
            newMatrix[0][i] = if (gradient.getDouble(i) < 0) {
                val max = elements.maxOrNull()!!.toDouble()
                if (max > 0) random.nextDouble(max, b * max).toFloat()
                else random.nextDouble(max, max / b).toFloat()
            } else {
                val min = elements.minOrNull()!!.toDouble()
                if (min > 0) random.nextDouble(min / b, min).toFloat()
                else random.nextDouble(b * min, min).toFloat()
            }
        }
        return NDArray(newMatrix)
    }
}
