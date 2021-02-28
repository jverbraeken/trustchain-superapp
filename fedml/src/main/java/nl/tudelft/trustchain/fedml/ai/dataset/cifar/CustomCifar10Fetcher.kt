package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.dataset.CustomFileSplit
import org.apache.commons.io.FileUtils
import org.datavec.image.transform.ImageTransform
import org.deeplearning4j.common.resources.DL4JResources
import org.deeplearning4j.common.resources.ResourceType
import org.nd4j.common.base.Preconditions
import org.nd4j.common.util.ArchiveUtils
import org.nd4j.linalg.dataset.DataSet
import java.io.File
import java.io.IOException
import java.net.URL
import kotlin.math.min

private val logger = KotlinLogging.logger("CustomCifar10Fetcher")

class CustomCifar10Fetcher {
    private fun downloadAndExtract(transfer: Boolean, localCache: File) {
        val tmpFile = File(System.getProperty("java.io.tmpdir"), if (transfer) LOCAL_FILE_NAME_TRANSFER else LOCAL_FILE_NAME_REGULAR)

        // Check empty cache
        if (localCache.exists()) {
            val list = localCache.listFiles()
            if (list == null || list.isEmpty()) localCache.delete()
        }
        if (!localCache.exists()) {
            localCache.mkdirs()
            tmpFile.delete()
            logger.info("Downloading dataset to " + tmpFile.absolutePath)
            FileUtils.copyURLToFile(URL(if (transfer) REMOTE_DATA_URL_TRANSFER else REMOTE_DATA_URL_REGULAR), tmpFile)
        } else {
            // Directory exists and is non-empty - assume OK
            logger.info("Using cached dataset at " + localCache.absolutePath)
            return
        }
        try {
            ArchiveUtils.unzipFileTo(tmpFile.absolutePath, localCache.absolutePath, false)
        } catch (t: Throwable) {
            // Catch any errors during extraction, and delete the directory to avoid leaving the dir in an invalid state
            if (localCache.exists()) {
                FileUtils.deleteDirectory(localCache)
            }
            throw t
        }
    }

    private fun deleteIfEmpty(localCache: File) {
        if (localCache.exists()) {
            val files = localCache.listFiles()
            if (files == null || files.isEmpty()) {
                try {
                    FileUtils.deleteDirectory(localCache)
                } catch (e: IOException) {
                    logger.debug("Error deleting directory: {}", localCache)
                }
            }
        }
    }

    @Synchronized
    fun getRecordReader(
        rngSeed: Long,
        imgDim: IntArray?,
        dataSetType: CustomDataSetType,
        imageTransform: ImageTransform?,
        iteratorDistribution: IntArray,
        maxSamples: Int,
        transfer: Boolean,
    ): CustomImageRecordReader {
        Preconditions.checkState(
            imgDim == null || imgDim.size == 2,
            "Invalid image dimensions: must be null or length 2. Got: %s",
            imgDim
        )
        // check empty cache
        val localCache = DL4JResources.getDirectory(ResourceType.DATASET, if (transfer) LOCAL_CACHE_NAME_TRANSFER else LOCAL_CACHE_NAME_REGULAR)
        deleteIfEmpty(localCache)
        try {
            if (!localCache.exists()) {
                logger.debug { "Creating cache..." }
                downloadAndExtract(transfer, localCache)
            }
        } catch (e: Exception) {
            throw RuntimeException("Could not download CIFAR-10", e)
        }

        val train = (dataSetType == CustomDataSetType.TRAIN)
        val datasetPath = if (train) File(localCache, "/train/") else File(localCache, "/test/")
        val random = kotlin.random.Random(rngSeed)
        val maxElementsPerLabel = if (transfer) (0 until NUM_LABELS_TRANSFER).map { if (train) NUM_TRAINING_SAMPLES_PER_LABEL_TRANSFER else 10 }.toIntArray() else iteratorDistribution.map { min(maxSamples, it) }.toIntArray()
        val numSamplesPerLabel = if (train) {
            if (transfer) NUM_TRAINING_SAMPLES_PER_LABEL_TRANSFER else NUM_TRAINING_SAMPLES_PER_LABEL_REGULAR
        } else {
            if (transfer) NUM_TESTING_SAMPLES_PER_LABEL_TRANSFER else NUM_TESTING_SAMPLES_PER_LABEL_REGULAR
        }
        val files = CustomFileSplit(datasetPath, random, numSamplesPerLabel, if (transfer) NUM_LABELS_TRANSFER else NUM_LABELS_REGULAR).files
        val fileSelection = files
            .mapIndexed { i, list -> list.copyOfRange(0, maxElementsPerLabel[i]) }
            .flatMap { it.asIterable() }
            .shuffled(random)
            .toTypedArray()
        val h = imgDim?.get(0) ?: INPUT_HEIGHT
        val w = imgDim?.get(1) ?: INPUT_WIDTH
        val labels = (0 until if (transfer) NUM_LABELS_TRANSFER else NUM_LABELS_REGULAR).map { it.toString() }.toTypedArray()
        val testBatches = if (dataSetType == CustomDataSetType.FULL_TEST) {
            createTestBatches(
                h.toLong(),
                w.toLong(),
                imageTransform,
                files,
                transfer,
                labels,
            )
        } else null

        return CustomImageRecordReader(
            h.toLong(),
            w.toLong(),
            INPUT_CHANNELS.toLong(),
            imageTransform,
            fileSelection,
            false,
            testBatches,
            labels,
            dataSetType
        )
    }

    private fun createTestBatches(
        h: Long,
        w: Long,
        imageTransform: ImageTransform?,
        files: Array<out Array<File>>,
        transfer: Boolean,
        labels: Array<String>
    ): Array<DataSet?> {
        return files.map {
            if (files.isEmpty()) null
            else {
                CustomRecordReaderDataSetIterator(
                    CustomImageRecordReader(
                        h,
                        w,
                        INPUT_CHANNELS.toLong(),
                        imageTransform,
                        it,
                        true,
                        null,
                        labels,
                        CustomDataSetType.TEST
                    ),
                    20,
                    1,
                    if (transfer) NUM_LABELS_TRANSFER else NUM_LABELS_REGULAR
                ).next(20)
            }
        }.toTypedArray()
    }

    companion object {
        private const val LOCAL_CACHE_NAME_TRANSFER = "cifar10"
        private const val LOCAL_CACHE_NAME_REGULAR = "cifar100"
        private const val REMOTE_DATA_URL_TRANSFER = "https://api.onedrive.com/v1.0/shares/s!AvNMRY4ml2WPgZpmYiFDOhNyFgakRQ/root/content"
        private const val REMOTE_DATA_URL_REGULAR = "https://api.onedrive.com/v1.0/shares/s!AvNMRY4ml2WPgaA98zlT5itmCF0s8w/root/content"
        private const val LOCAL_FILE_NAME_TRANSFER = "cifar10_dl4j.v1.zip"
        private const val LOCAL_FILE_NAME_REGULAR = "cifar100_dl4j.v1.zip"
        const val INPUT_WIDTH = 32
        const val INPUT_HEIGHT = 32
        const val INPUT_CHANNELS = 3
        const val NUM_LABELS_TRANSFER = 10
        const val NUM_LABELS_REGULAR = 100
        const val NUM_TRAINING_SAMPLES_PER_LABEL_TRANSFER = 5000
        const val NUM_TRAINING_SAMPLES_PER_LABEL_REGULAR = 500
        const val NUM_TESTING_SAMPLES_PER_LABEL_TRANSFER = 1000
        const val NUM_TESTING_SAMPLES_PER_LABEL_REGULAR = 100
    }
}
