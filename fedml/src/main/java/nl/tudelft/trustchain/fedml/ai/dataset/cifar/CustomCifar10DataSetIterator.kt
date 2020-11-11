package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.fetchers.Cifar10Fetcher
import org.deeplearning4j.datasets.fetchers.DataSetType


class CustomCifar10DataSetIterator(
    iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: DataSetType,
    behavior: Behaviors
) : RecordReaderDataSetIterator(
    CustomCifar10Fetcher(
        iteratorConfiguration.distribution,
        iteratorConfiguration.maxTestSamples?.value ?: Integer.MAX_VALUE
    ).getRecordReader(seed, null, dataSetType, null),
    iteratorConfiguration.batchSize.value,
    1,
    Cifar10Fetcher.NUM_LABELS
)
