package nl.tudelft.trustchain.fedml.ai.dataset.har_unused

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import java.io.File


class HARDataSetIterator(
    iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDataSetType,
    baseDirectory: File,
    behavior: Behaviors,
    transfer: Boolean,
) : CustomBaseDatasetIterator(
    iteratorConfiguration.batchSize.value,
    -1,
    HARDataFetcher(
        baseDirectory,
        seed,
        iteratorConfiguration.distribution.toIntArray(),
        dataSetType,
        if (dataSetType == CustomDataSetType.TEST || dataSetType == CustomDataSetType.FULL_TEST) iteratorConfiguration.maxTestSamples.value else Integer.MAX_VALUE,
        behavior,
        transfer
    )
), CustomDataSetIterator {
    override val testBatches by lazy {customFetcher.testBatches}

    override fun getLabels(): List<String> {
        return (fetcher as HARDataFetcher).labels
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: CustomDataSetType,
            baseDirectory: File,
            behavior: Behaviors,
            transfer: Boolean,
        ): HARDataSetIterator {
            return HARDataSetIterator(iteratorConfiguration, seed, dataSetType, baseDirectory, behavior, transfer)
        }
    }
}

