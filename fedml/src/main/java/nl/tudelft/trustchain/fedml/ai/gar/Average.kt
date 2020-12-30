package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

private val logger = KotlinLogging.logger("Average")

class Average : AggregationRule() {
    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
        logging: Boolean
    ): INDArray {
        debug(logging) { formatName("Simple average") }
        val models = HashMap<Int, INDArray>()
        models[-1] = oldModel.sub(gradient)
        models.putAll(newOtherModels)
        debug(logging) { "Found ${models.size} models in total" }
        return average(models)
    }

    override fun isDirectIntegration(): Boolean {
        return false
    }

    private fun average(models: HashMap<Int, INDArray>) : INDArray {
        var arr: INDArray? = null
        models.onEachIndexed { indexAsNum, (_, model) ->
            if (indexAsNum == 0) {
                arr = model.dup()
            } else {
                arr!!.addi(model)
            }
        }
        return arr!!.divi(models.size)
    }
}
