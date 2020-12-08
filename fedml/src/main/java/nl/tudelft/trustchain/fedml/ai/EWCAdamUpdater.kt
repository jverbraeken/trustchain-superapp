package nl.tudelft.trustchain.fedml.ai

import mu.KotlinLogging
import org.apache.commons.math3.util.FastMath
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.shape.Shape
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.learning.GradientUpdater
import org.nd4j.linalg.ops.transforms.Transforms
import java.util.*

private val logger = KotlinLogging.logger("EWCAdamUpdater")

class EWCAdamUpdater(private val config: EWCAdam) : GradientUpdater<EWCAdam> {
    private var m: INDArray? = null
    private var v: INDArray? = null
    private var gradientReshapeOrder = 0.toChar()

    override fun setState(stateMap: Map<String, INDArray>, initialize: Boolean) {
        check(!(!stateMap.containsKey(M_STATE) || !stateMap.containsKey(V_STATE) || stateMap.size != 2)) { "State map should contain only keys [" + M_STATE + "," + V_STATE + "] but has keys " + stateMap.keys }
        m = stateMap[M_STATE]
        v = stateMap[V_STATE]
    }

    override fun getState(): Map<String, INDArray> {
        val r: MutableMap<String, INDArray> = HashMap()
        r[M_STATE] = m!!
        r[V_STATE] = v!!
        return r
    }

    override fun setStateViewArray(viewArray: INDArray, gradientShape: LongArray, gradientOrder: Char, initialize: Boolean) {
        require(viewArray.isRowVector) { "Invalid input: expect row vector input" }
        if (initialize) viewArray.assign(0)
        val length = viewArray.length()
        m = viewArray[NDArrayIndex.point(0), NDArrayIndex.interval(0, length / 2)]
        v = viewArray[NDArrayIndex.point(0), NDArrayIndex.interval(length / 2, length)]

        //Reshape to match the expected shape of the input gradient arrays
        m = Shape.newShapeNoCopy(m, gradientShape, gradientOrder == 'f')
        v = Shape.newShapeNoCopy(v, gradientShape, gradientOrder == 'f')
        check(!(m == null || v == null)) { "Could not correctly reshape gradient view arrays" }
        gradientReshapeOrder = gradientOrder
    }

    /**
     * Calculate the update based on the given gradient
     *
     * @param gradient  the gradient to get the update for
     * @param iteration
     * @return the gradient
     */
    override fun applyUpdater(gradient: INDArray, iteration: Int, epoch: Int) {
//        val penalty = 1000 * LossEWC.fishers!!.mul(LossEWC.model.params().sub(LossEWC.old_var_list!!).mul(LossEWC.model.params().sub(
//            LossEWC.old_var_list!!))).sumNumber().toDouble()

        check(!(m == null || v == null)) { "Updater has not been initialized with view state" }
        val beta1 = config.beta1
        val beta2 = config.beta2
        val learningRate = config.getLearningRate(iteration, epoch)
        val epsilon = config.epsilon

        val oneMinusBeta1Grad = gradient.mul(1.0 - beta1)
        m!!.muli(beta1).addi(oneMinusBeta1Grad)

        val oneMinusBeta2GradSquared = gradient.mul(gradient).muli(1 - beta2)
        v!!.muli(beta2).addi(oneMinusBeta2GradSquared)

        val beta1t = FastMath.pow(beta1, iteration + 1)
        val beta2t = FastMath.pow(beta2, iteration + 1)

        var alphat = learningRate * FastMath.sqrt(1 - beta2t) / (1 - beta1t)
        if (java.lang.Double.isNaN(alphat) || alphat == 0.0) alphat = epsilon
        val sqrtV = Transforms.sqrt(v!!.dup(gradientReshapeOrder), false).addi(epsilon)
        val tmp = m!!.mul(alphat).divi(sqrtV)

//        val diffSquared = diff.mul(diff)
//        val penalty = fisher.mul(diffSquared)

        gradient.assign(
            if (fisher == null || old_var_list == null) tmp
            else {
                val ones = NDArray(1, fisher!![0].size(1).toInt(), Nd4j.order()).addi(1)
                val bufferv = NDArray(1, fisher!![0].size(1).toInt(), Nd4j.order())
                /*val bufferw = NDArray(1, fisher!![0].size(1).toInt(), Nd4j.order())
                for (arr in fisher!!.zip(old_var_list!!)) {
                    val b = arr.first.div(arr.first.maxNumber())
                    val a = b.replaceWhere(b.mul(-1), Conditions.lessThan(0.0))*//*.replaceWhere(ones, Conditions.greaterThan(1.0))*//*
//                    val c = ones.sub(ones.sub(a).mul(ones.sub(a))).mul(0.5)
                    bufferw.addi(a)
//                val a = fisher!!.div(fisher!!.maxNumber()).mul(diff)
                }
                a.addi(1e-20)*/
                for (arr in fisher!!.zip(old_var_list!!)) {
                    val diff = model.params().sub(arr.second)!!.add(tmp)
                    val b = arr.first.div(arr.first.maxNumber())
                    val a = b.replaceWhere(b.mul(-1), Conditions.lessThan(0.0))/*.replaceWhere(ones, Conditions.greaterThan(1.0))*/
//                    val c = ones.sub(ones.sub(a).mul(ones.sub(a))).mul(0.5)
                    bufferv.addi(a/*.div(bufferw)*/.mul(diff))
//                val a = fisher!!.div(fisher!!.maxNumber()).mul(diff)
                }
                bufferv.divi(fisher!!.size)//muli(bufferw.replaceWhere(ones, Conditions.greaterThan(1.0)))
                tmp.addi(bufferv)
                tmp

//                tmp.add(diff)
//                val a = fisher!!.mul(diff.mul(diff))
//                if (a.maxNumber().toDouble() == 0.0) tmp
//                else tmp.mul(fisher!!.div(fisher!!.maxNumber()).mul(diff))
//                else tmp.add(a.mul(1.0 / a.maxNumber().toDouble()))
//                val ones = NDArray(fisher!!.size(1).toInt(), 1, Nd4j.order()).addi(1)
//                 tmp.div(fisher!!.mul(1e30).mul(diff.mul(diff)).replaceWhere(ones, Conditions.lessThan(1.0)))

            }
        )
    }

    companion object {
        var fisher: List<INDArray>? = null
        var old_var_list: List<INDArray>? = null
        lateinit var model: CustomMultiLayerNetwork
        const val M_STATE = "M"
        const val V_STATE = "V"
    }

    override fun getConfig(): EWCAdam {
        return config
    }
}
