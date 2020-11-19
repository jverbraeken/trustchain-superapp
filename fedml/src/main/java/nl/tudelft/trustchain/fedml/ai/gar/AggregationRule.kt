package nl.tudelft.trustchain.fedml.ai.gar

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

abstract class AggregationRule {
    abstract fun integrateParameters(
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator,
        allOtherModelsBuffer: ArrayDeque<Pair<Int, INDArray>>,
        logging: Boolean,
        testBatches: List<DataSet?>,
        countPerPeer: Map<Int, Int>
    ): INDArray

    abstract fun isDirectIntegration(): Boolean

    protected fun formatName(name: String): String {
        return "<====      $name      ====>"
    }
}
