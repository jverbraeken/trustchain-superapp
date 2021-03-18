package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.NumAttackers
import nl.tudelft.trustchain.fedml.ai.gar.getKrum
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.shape.Shape
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
        otherModels: Map<Int, INDArray>,
        random: Random
    ): Map<Int, INDArray> {
        logger.debug { formatName("Fang 2020 Krum") }
        val models = arrayOf<INDArray>(oldModel.sub(gradient), *(otherModels.values.toTypedArray()))
        logger.debug { "Found ${models.size} models in total" }
        if (models.size < 4) {
            logger.debug { "Too few models => no attack vectors generated" }
            return mapOf()
        }
        val modelsAsArrays = models.map { toFloatArray(it) }.toTypedArray()

        // w1
        val s = Array(modelsAsArrays[0].size) { FloatArray(modelsAsArrays[0][0].size) }
        val fm = toFloatArray(gradient)
        for (i in fm.indices) {
            for (j in fm[0].indices) {
                s[i][j] = if (fm[i][j] < 0) -1f else 1f
            }
        }
        val ns = NDArray(s)

        val d = modelsAsArrays[0].size.toFloat()
        val m = otherModels.size + numAttackers.num
        val c = numAttackers.num
        var lambda = (1.0 / ((m - 2 * c - 1) * sqrt(d))) *
            otherModels.values
                .map { a -> otherModels.values.map { it.distance2(a) }.sum() }.toTypedArray()
                .minOrNull()!!
                .toFloat() + (1.0 / sqrt(d)) *
            otherModels.values
                .map { it.distance2(oldModel) }.toTypedArray()
                .maxOrNull()!!
//        logger.debug { "lambda 1: $lambda" }

        while (lambda >= 1e-5) {
            val w1 = oldModel.sub(ns.mul(lambda))
            // TODO add noise eta
            val newModels = Array<INDArray>(c) { w1 }
            val combinedModels = arrayOf(*models, *newModels)
            if (getKrum(combinedModels, b) >= models.size) {
                logger.debug { "${newModels.map { it.minNumber() }}; ${newModels.map { it.maxNumber() }}" }
                return transformToResult(newModels.map {
                    if (it.minNumber().toDouble() < -10 || it.maxNumber().toDouble() > 10) oldModel else it
                }.toTypedArray())
            }
            lambda /= 2
        }
//        logger.debug { "lambda 2: $lambda" }
        val w1 = oldModel.sub(ns.mul(lambda))
        val newModels = Array<INDArray>(c) { w1 }

//        logger.debug { "${newModels.map { it.minNumber() }}; ${newModels.map { it.maxNumber() }}" }
        return transformToResult(newModels)
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
}
