package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.fetchers.Cifar10Fetcher
import org.deeplearning4j.datasets.fetchers.DataSetType
import java.io.File


class CustomCifar10DataSetIterator(
    private val iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: DataSetType,
    behavior: Behaviors
) : RecordReaderDataSetIterator(
    CustomCifar10Fetcher(
        iteratorConfiguration.distribution,
        iteratorConfiguration.maxTestSamples?.value ?: Integer.MAX_VALUE,
        behavior
    ).getRecordReader(seed, null, dataSetType, null),
    iteratorConfiguration.batchSize.value,
    1,
    Cifar10Fetcher.NUM_LABELS
) {
    override fun getLabels(): List<String> {
        return iteratorConfiguration.distribution.value.filter { it > 0 }.map { it.toString() }.toCollection(ArrayList())
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: DataSetType,
            baseDirectory: File,
            behavior: Behaviors
        ): CustomCifar10DataSetIterator {
            return CustomCifar10DataSetIterator(iteratorConfiguration, seed, dataSetType, behavior)
        }
    }
}
