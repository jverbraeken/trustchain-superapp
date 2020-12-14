package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.concurrent.ConcurrentLinkedDeque

private val logger = KotlinLogging.logger("Average")

class NoAveraging : AggregationRule() {
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
        debug(logging) { formatName("No averaging") }
        return oldModel.sub(gradient)
    }

    override fun isDirectIntegration(): Boolean {
        return false
    }
}
