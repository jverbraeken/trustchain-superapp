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
import java.net.URI
import java.net.URL
import java.util.zip.Adler32
import java.util.zip.Checksum
import kotlin.math.min

private val logger = KotlinLogging.logger("CustomCifar10Fetcher")

class CustomCifar10Fetcher {
    private fun downloadAndExtract() {
        val localFilename = "cifar10_dl4j.v1.zip"
        val tmpFile = File(System.getProperty("java.io.tmpdir"), localFilename)
        val localCacheDir = getLocalCacheDir()

        // Check empty cache
        if (localCacheDir.exists()) {
            val list = localCacheDir.listFiles()
            if (list == null || list.isEmpty()) localCacheDir.delete()
        }
        if (!localCacheDir.exists()) {
            localCacheDir.mkdirs()
            tmpFile.delete()
            logger.info("Downloading dataset to " + tmpFile.absolutePath)
            FileUtils.copyURLToFile(URL(REMOTE_DATA_URL), tmpFile)
        } else {
            // Directory exists and is non-empty - assume OK
            logger.info("Using cached dataset at " + localCacheDir.absolutePath)
            return
        }
        logger.info("Verifying download...")
        val adler: Checksum = Adler32()
        FileUtils.checksum(tmpFile, adler)
        val localChecksum = adler.value
        logger.info("Checksum local is $localChecksum, expecting $EXPECTED_CHECKSUM")
        if (localChecksum != EXPECTED_CHECKSUM) {
            logger.error("Checksums do not match. Cleaning up files and failing...")
            tmpFile.delete()
            throw IllegalStateException("Dataset file failed checksum: $tmpFile - expected checksum $EXPECTED_CHECKSUM vs. actual checksum $localChecksum. If this error persists, please open an issue at https://github.com/deeplearning4j/deeplearning4j.")
        }
        try {
            ArchiveUtils.unzipFileTo(tmpFile.absolutePath, localCacheDir.absolutePath, false)
        } catch (t: Throwable) {
            // Catch any errors during extraction, and delete the directory to avoid leaving the dir in an invalid state
            if (localCacheDir.exists()) FileUtils.deleteDirectory(localCacheDir)
            throw t
        }
    }

    private fun getLocalCacheDir(): File {
        return DL4JResources.getDirectory(ResourceType.DATASET, LOCAL_CACHE_NAME)
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
        set: CustomDataSetType,
        imageTransform: ImageTransform?,
        iteratorDistribution: IntArray,
        maxSamples: Int,
    ): CustomImageRecordReader {
        Preconditions.checkState(
            imgDim == null || imgDim.size == 2,
            "Invalid image dimensions: must be null or length 2. Got: %s",
            imgDim
        )
        // check empty cache
        val localCache = getLocalCacheDir()
        deleteIfEmpty(localCache)
        try {
            if (!localCache.exists()) {
                logger.debug { "Creating cache..." }
                downloadAndExtract()
            }
        } catch (e: Exception) {
            throw RuntimeException("Could not download CIFAR-10", e)
        }

        val train = (set == CustomDataSetType.TRAIN)
        val datasetPath = if (train) File(localCache, "/train/") else File(localCache, "/test/")
        val random = kotlin.random.Random(rngSeed)
        val maxElementsPerLabel = iteratorDistribution.map { min(maxSamples, it) }
        val numSamplesPerLabel = if (train) NUM_TRAINING_SAMPLES_PER_LABEL else NUM_TESTING_SAMPLES_PER_LABEL
        val files = CustomFileSplit(datasetPath, random, numSamplesPerLabel).files
        val fileSelection = files
            .mapIndexed { i, list -> list.copyOfRange(0, maxElementsPerLabel[i]) }
            .flatMap { it.asIterable() }
            .shuffled(random)
            .toTypedArray()
        val h = imgDim?.get(0) ?: INPUT_HEIGHT
        val w = imgDim?.get(1) ?: INPUT_WIDTH
        val testBatches = if (set == CustomDataSetType.FULL_TEST) {
            createTestBatches(
                h.toLong(),
                w.toLong(),
                imageTransform,
                files
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
        )
    }

    private fun createTestBatches(
        h: Long,
        w: Long,
        imageTransform: ImageTransform?,
        files: Array<out Array<File>>
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
                        null
                    ),
                    20,
                    1,
                    NUM_LABELS
                ).next(20)
            }
        }.toTypedArray()
    }

    companion object {
        private const val LOCAL_CACHE_NAME = "cifar10"
        private const val EXPECTED_CHECKSUM = 292852033L
        private const val REMOTE_DATA_URL = "https://api.onedrive.com/v1.0/shares/s!AvNMRY4ml2WPgZpmYiFDOhNyFgakRQ/root/content"
        const val INPUT_WIDTH = 32
        const val INPUT_HEIGHT = 32
        const val INPUT_CHANNELS = 3
        const val NUM_LABELS = 10
        const val NUM_TRAINING_SAMPLES_PER_LABEL = 5000
        const val NUM_TESTING_SAMPLES_PER_LABEL = 1000
    }
}
