package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.NumAttackers
import nl.tudelft.trustchain.fedml.ai.gar.getKrum
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import kotlin.math.sqrt
import kotlin.random.Random

private val logger = KotlinLogging.logger("Fang2020Krum")

/**
 * IMPORTANT: b refers in this code and the original paper NOT to the amount of attackers, but to the attacker's range
 */
class Fang2020Krum(private val b: Int) : ModelPoisoningAttack() {
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

        // w1
        val s = Array(1) { FloatArray(modelsAsArrays[0].size) }
        val fm = gradient.toFloatMatrix()[0]
        for (i in fm.indices) {
            s[0][i] = if (fm[i] < 0) -1f else 1f
        }
        val ns = NDArray(s)

        val d = modelsAsArrays[0].size.toFloat()
        val m = otherModels.size + numAttackers.num
        val c = numAttackers.num
        var lambda = (1.0 / ((m - 2 * c - 1) * sqrt(d))) *
            otherModels.map { a -> otherModels.map { it.distance2(a) }.sum() }.min()!!.toFloat() +
            (1.0 / sqrt(d)) * otherModels.map { it.distance2(oldModel) }.max()!!

        while (lambda >= 1e-5) {
            val w1 = oldModel.sub(ns.mul(lambda))
            val newModels = arrayListOf<INDArray>()
            // TODO add noise eta
            for (i in 0 until c) {
                newModels.add(w1)
            }
            val combinedModels = ArrayList(models)
            combinedModels.addAll(newModels)
            if (getKrum(combinedModels, b) >= models.size) {
                return newModels
            }
            lambda /= 2
        }
        val w1 = oldModel.sub(ns.mul(lambda))
        val newModels = arrayListOf<INDArray>()
        for (i in 0 until c) {
            newModels.add(w1)
        }
        return newModels
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
