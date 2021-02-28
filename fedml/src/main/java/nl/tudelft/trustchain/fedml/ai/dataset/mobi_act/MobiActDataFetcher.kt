package nl.tudelft.trustchain.fedml.ai.dataset.mobi_act

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDataFetcher
import org.apache.commons.io.FilenameUtils
import org.apache.commons.io.FileUtils
import org.deeplearning4j.common.resources.DL4JResources
import org.deeplearning4j.common.resources.ResourceType
import org.nd4j.common.util.ArchiveUtils
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.File
import java.io.IOException
import java.net.URL
import java.util.stream.IntStream

private val logger = KotlinLogging.logger("MobiActDataFetcher")

class MobiActDataFetcher(
    val iteratorDistribution: IntArray,
    seed: Long,
    val dataSetType: CustomDataSetType,
    maxTestSamples: Int,
    behavior: Behaviors,
    transfer: Boolean,
) : CustomBaseDataFetcher(seed) {
    override val testBatches by lazy { createTestBatches() }

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
            logger.info("Unzipping file")
            ArchiveUtils.unzipFileTo(tmpFile.absolutePath, localCache.absolutePath, false)
        } catch (t: Throwable) {
            // Catch any errors during extraction, and delete the directory to avoid leaving the dir in an invalid state
            if (localCache.exists()) {
                FileUtils.deleteDirectory(localCache)
            }
            throw t
        }
    }

    private var featureData = Array(1) { Array(500) { doubleArrayOf() } }

    init {
        if (shuffledSamplesTrain == null) {
            // check empty cache
            val localCache = DL4JResources.getDirectory(
                ResourceType.DATASET,
                if (transfer) LOCAL_CACHE_NAME_TRANSFER else LOCAL_CACHE_NAME_REGULAR
            )
            deleteIfEmpty(localCache)
            try {
                if (!localCache.exists()) {
                    logger.debug { "Creating cache..." }
                    downloadAndExtract(transfer, localCache)
                }
            } catch (e: Exception) {
                throw RuntimeException("Could not download CIFAR-10", e)
            }

            val root = DL4JResources.getDirectory(
                ResourceType.DATASET,
                if (transfer) LOCAL_CACHE_NAME_TRANSFER else LOCAL_CACHE_NAME_REGULAR
            ).absolutePath
            val file = if (transfer) {
                File(FilenameUtils.concat(root, "WISDM_ar_v1.1_raw.txt"))
            } else {
                File(FilenameUtils.concat(root, "WISDM_ar_v1.1_raw.txt"))
            }
            val lines = file.bufferedReader().readLines()
            logger.debug { "1" }
            val elements = lines.map { it.fastsplit(',') }
            logger.debug { "2" }
            activityToSamples = elements
                .groupBy { it.first }
                .map { (label, samples) ->
                    Pair(label, samples.groupBy { it.second }.map { it.value.subList(0, 50).map { it.third }.toTypedArray() })
                }
            val allSamples = activityToSamples!!
                .map { (label, samplesPerHuman) ->
                    samplesPerHuman.map { Pair(label, it) }
                }
                .flatten()
                /*.map { (label, data) ->
                    val newData = data.copyOf(500).map { it ?: doubleArrayOf(0.0, 0.0, 0.0) }.toTypedArray()
                    Pair(label, newData)
                }*/
                .shuffled()
            logger.debug { "4" }

            shuffledSamplesTrain = allSamples.subList(0, (0.7 * allSamples.size).toInt()).toTypedArray()
            shuffledSamplesTest = allSamples.subList((0.7 * allSamples.size).toInt(), allSamples.size).toTypedArray()
        }

        totalExamples = if (dataSetType == CustomDataSetType.TRAIN) shuffledSamplesTrain!!.size else shuffledSamplesTest!!.size
        numOutcomes = if (transfer) activityToSamples!!.size else -10
        cursor = 0
        inputColumns = 500
        order = IntStream.range(0, totalExamples).toArray()
        reset() //Shuffle order
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

    override fun fetch(numExamples: Int) {
        check(hasMore()) { "Unable to get more; there are no more images" }
        var labels = Nd4j.zeros(DataType.FLOAT, numExamples.toLong(), 6)
        if (featureData.size != numExamples) {
            featureData = Array(numExamples) { Array(3) { DoubleArray(500) } }
        }
        var actualExamples = 0
        for (i in 0 until numExamples) {
            if (!hasMore()) break
            val (label, data) = if (dataSetType == CustomDataSetType.TRAIN) {
                val a = order[cursor]
                Pair(shuffledSamplesTrain!![a].first, transpose(shuffledSamplesTrain!![a].second))
            } else {
                val a = order[cursor]
                Pair(shuffledSamplesTest!![a].first, transpose(shuffledSamplesTest!![a].second))
            }
            featureData[actualExamples] = data
            labels.put(actualExamples, label, 1.0f)
            actualExamples++
            cursor++
        }
        val features = Nd4j.create(
            if (featureData.size == actualExamples) featureData
            else featureData.copyOfRange(0, actualExamples)
        )
        if (actualExamples < numExamples) {
            labels = labels[NDArrayIndex.interval(0, actualExamples), NDArrayIndex.all()]
        }
        curr = DataSet(features, labels)
        if (dataSetType != CustomDataSetType.TRAIN) {
            cursor = totalExamples
        }
    }

    private fun transpose(second: Array<DoubleArray>): Array<DoubleArray> {
        val result = Array(second[0].size) { DoubleArray(second.size) }
        for (i in second.indices) {
            for (j in second[i].indices) {
                result[j][i] = second[i][j]
            }
        }
        return result
    }

    private fun createTestBatches(): Array<DataSet?> {
        /*val testBatches = man.createTestBatches()
        if (featureData.size != TEST_BATCH_SIZE) {
            featureData = Array(TEST_BATCH_SIZE) { FloatArray(SIZE_IMAGE) }
        }
        val result = arrayListOf<DataSet?>()
        for ((label, batch) in testBatches.withIndex()) {
            result.add(
                if (batch.isEmpty()) null
                else createTestBatch(label, batch)
            )
        }
        return result.toTypedArray()*/
        return arrayOf()
    }

    /*private fun createTestBatch(label: Int, batch: Array<FloatArray>): DataSet {
        val numSamplesInBatch = batch.size
        val labels = Nd4j.zeros(DataType.FLOAT, numSamplesInBatch.toLong(), numOutcomes.toLong())
        for ((i, img) in batch.withIndex()) {
            labels.put(i, label, 1.0f)
            featureData[i] = img
        }
        val features = Nd4j.create(featureData)
        features.divi(255.0)
        return DataSet(features, labels)
    }*/

    val labels: List<String>
        get() = listOf()//man.getLabels()

    companion object {
        const val TEST_BATCH_SIZE = 10
        private const val REMOTE_DATA_URL_REGULAR = "https://api.onedrive.com/v1.0/shares/s!AvNMRY4ml2WPgaBQ2on0dU-zWnGL9A/root/content"
        private const val REMOTE_DATA_URL_TRANSFER = "https://api.onedrive.com/v1.0/shares/s!AvNMRY4ml2WPgaBQ2on0dU-zWnGL9A/root/content"
        private const val LOCAL_CACHE_NAME_REGULAR = "mobi_act"
        private const val LOCAL_CACHE_NAME_TRANSFER = "wisdm"
        private const val LOCAL_FILE_NAME_REGULAR = "?"
        private const val LOCAL_FILE_NAME_TRANSFER = "WISDM_ar_v1.1_raw.zip"

        private var activityToSamples: List<Pair<Int, List<Array<DoubleArray>>>>? = null
        private var shuffledSamplesTrain: Array<Pair<Int, Array<DoubleArray>>>? = null
        private var shuffledSamplesTest: Array<Pair<Int, Array<DoubleArray>>>? = null
    }
}

private fun String.fastsplit(c: Char): Triple<Int, Int, DoubleArray> {
    /*val labels = arrayOf(
        "Walking",
        "Jogging",
        "Upstairs",
        "Downstairs",
        "Sitting",
        "Standing"
    )*/
    var activity = 0
    var human = 0
    val data = DoubleArray(3)
    var start = 0
    var count = 0
    try {
        for (i in this.indices) {
            if (this[i] == c) {
                when {
                    count == 0 -> {
                        human = this.substring(start, i).toInt()
                    }
                    count == 1 -> {
                        activity = when (this[start + 2]) {
                            'l' -> 0
                            'g' -> 1
                            's' -> 2
                            'w' -> 3
                            't' -> 4
                            'a' -> 5
                            else -> throw RuntimeException("Impossible")
                        }
                    }
                    count >= 3 -> {
                        data[count - 3] = this.substring(start, i).toDouble()
                    }
                }
                start = i + 1
                count++
            }
        }
    } catch (e: Exception) {
        logger.debug { this }
    }
    try {
        data[2] = this.substring(start).toDouble()
    } catch (e: Exception) {
        logger.debug { start }
        logger.debug { this }
        logger.debug { count }
    }
    return Triple(activity, human, data)
}
