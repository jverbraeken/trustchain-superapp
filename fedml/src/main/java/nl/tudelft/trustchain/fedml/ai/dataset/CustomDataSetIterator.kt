package nl.tudelft.trustchain.fedml.ai.dataset

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

interface CustomDataSetIterator : DataSetIterator {
    val testBatches: Array<DataSet?>
}
