package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.NumAttackers
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
        otherModels: List<INDArray>,
        random: Random
    ): Collection<INDArray> {
        logger.debug { formatName("Fang 2020 Trimmed Mean") }
        val models: MutableList<INDArray> = arrayListOf(oldModel.sub(gradient))
        otherModels.forEach { models.add(it) }
        logger.debug { "Found ${models.size} models in total" }
        val modelsAsArrays = models.map { it.toFloatMatrix()[0] }
        val result = arrayListOf<INDArray>()
        for (i in 0 until numAttackers.num) {
            result.add(generateAttackVector(modelsAsArrays, gradient, random, b))
        }
        return result
    }

    private fun generateAttackVector(
        modelsAsArrays: List<FloatArray>,
        gradient: INDArray,
        random: Random,
        b: Int
    ): INDArray {
        val newMatrix = Array(1) { FloatArray(modelsAsArrays[0].size) }
        for (i in modelsAsArrays[0].indices) {
            val elements = ArrayList<Float>(modelsAsArrays.size)
            modelsAsArrays.forEach { elements.add(it[i]) }
            newMatrix[0][i] = if (gradient.getDouble(i) < 0) {
                val max = elements.max()!!.toDouble()
                if (max > 0) random.nextDouble(max, b * max).toFloat()
                else random.nextDouble(max, max / b).toFloat()
            } else {
                val min = elements.min()!!.toDouble()
                if (min > 0) random.nextDouble(min / b, min).toFloat()
                else random.nextDouble(b * min, min).toFloat()
            }
        }
        return NDArray(newMatrix)
    }
}
