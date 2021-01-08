package nl.tudelft.trustchain.fedml.ai.dataset.covid

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.har.HARDataFetcher
import org.nd4j.linalg.dataset.DataSet
import java.io.File


class COVIDDataSetIterator(
    iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDataSetType,
    baseDirectory: File,
    behavior: Behaviors
) : CustomBaseDatasetIterator(
    iteratorConfiguration.batchSize.value,
    -1,
    COVIDDataFetcher(
        baseDirectory,
        seed,
        iteratorConfiguration.distribution.toIntArray(),
        dataSetType,
        if (dataSetType == CustomDataSetType.TEST || dataSetType == CustomDataSetType.FULL_TEST) iteratorConfiguration.maxTestSamples.value else Integer.MAX_VALUE,
        behavior
    )
), CustomDataSetIterator {
    override val testBatches by lazy { arrayOf<DataSet?>() }

    override fun getLabels(): List<String> {
        return (fetcher as HARDataFetcher).labels
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: CustomDataSetType,
            baseDirectory: File,
            behavior: Behaviors
        ): COVIDDataSetIterator {
            return COVIDDataSetIterator(iteratorConfiguration, seed, dataSetType, baseDirectory, behavior)
        }
    }
}

