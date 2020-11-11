package nl.tudelft.trustchain.fedml.ai.dataset.har

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator
import java.io.File


class HARIterator(
    iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Int,
    dataSetType: DataSetType,
    baseDirectory: File,
    behavior: Behaviors
) : BaseDatasetIterator(
    iteratorConfiguration.batchSize.value,
    -1,
    HARDataFetcher(
        baseDirectory,
        seed,
        iteratorConfiguration.distribution.value,
        dataSetType,
        iteratorConfiguration.maxTestSamples?.value ?: Integer.MAX_VALUE
    )
)
