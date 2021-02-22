package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import java.io.File

private val logger = KotlinLogging.logger("CustomCifar10DataSetIterator")

class CustomCifar10DataSetIterator(
    iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDataSetType,
    behavior: Behaviors,
    transfer: Boolean,
) : CustomRecordReaderDataSetIterator(
    CustomCifar10Fetcher().getRecordReader(
        seed,
        null,
        dataSetType,
        null,
        iteratorConfiguration.distribution.toIntArray(),
        if (dataSetType == CustomDataSetType.TEST || dataSetType == CustomDataSetType.FULL_TEST) iteratorConfiguration.maxTestSamples.value else Integer.MAX_VALUE,
        transfer,
    ),
    iteratorConfiguration.batchSize.value,
    1,
    if (transfer) CustomCifar10Fetcher.NUM_LABELS_TRANSFER else CustomCifar10Fetcher.NUM_LABELS_REGULAR
), CustomDataSetIterator {
    override val testBatches by lazy { recordReader.testBatches!! }

    override fun getLabels(): List<String> {
        return recordReader.labels.toList()
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: CustomDataSetType,
            baseDirectory: File,
            behavior: Behaviors,
            transfer: Boolean,
        ): CustomCifar10DataSetIterator {
            return CustomCifar10DataSetIterator(iteratorConfiguration, seed, dataSetType, behavior, transfer)
        }
    }
}
