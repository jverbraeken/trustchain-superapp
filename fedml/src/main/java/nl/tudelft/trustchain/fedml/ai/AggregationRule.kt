package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

abstract class AggregationRule {
    abstract fun integrateParameters(
        myModel: Pair<INDArray, Int>,
        otherModelPairs: List<Pair<INDArray, Int>>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): Pair<INDArray, Int>
}
