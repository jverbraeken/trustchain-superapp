package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.nd4j.linalg.dataset.DataSet
import java.io.File

private val logger = KotlinLogging.logger("CustomCifar10DataSetIterator")

class CustomCifar10DataSetIterator(
    iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDataSetType,
    behavior: Behaviors,
) : CustomRecordReaderDataSetIterator(
    CustomCifar10Fetcher().getRecordReader(seed,
        null,
        dataSetType,
        null,
        iteratorConfiguration.distribution,
        if (dataSetType == CustomDataSetType.TEST || dataSetType == CustomDataSetType.FULL_TEST) iteratorConfiguration.maxTestSamples.value else Integer.MAX_VALUE),
    iteratorConfiguration.batchSize.value,
    1,
    CustomCifar10Fetcher.NUM_LABELS
), CustomDataSetIterator {
    override val testBatches by lazy { listOf<DataSet?>() }

    override fun getLabels(): List<String> {
        return (recordReader as CustomImageRecordReader).getUniqueLabels().toList()
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: CustomDataSetType,
            baseDirectory: File,
            behavior: Behaviors,
        ): CustomCifar10DataSetIterator {
            logger.debug { "Creating CustomCifar10DataSetIterator" }
            return CustomCifar10DataSetIterator(iteratorConfiguration, seed, dataSetType, behavior)
        }
    }
}
