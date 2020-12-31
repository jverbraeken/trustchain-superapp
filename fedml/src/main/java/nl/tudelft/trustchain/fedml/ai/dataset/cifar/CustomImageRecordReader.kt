package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import mu.KotlinLogging
import org.datavec.api.conf.Configuration
import org.datavec.api.records.Record
import org.datavec.api.records.metadata.RecordMetaData
import org.datavec.api.records.metadata.RecordMetaDataURI
import org.datavec.api.records.reader.BaseRecordReader
import org.datavec.api.split.InputSplit
import org.datavec.api.util.files.URIUtil
import org.datavec.api.util.ndarray.RecordConverter
import org.datavec.api.writable.IntWritable
import org.datavec.api.writable.Writable
import org.datavec.api.writable.batch.NDArrayRecordBatch
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.BaseImageRecordReader
import org.datavec.image.transform.ImageTransform
import org.nd4j.linalg.api.concurrency.AffinityManager
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.shade.guava.base.Preconditions
import java.io.*
import java.net.URI
import java.util.*

private val logger = KotlinLogging.logger("CustomImageRecordReader")

/**
 * Specific for CIFAR-10
 */
class CustomImageRecordReader(
    private val height: Long,
    private val width: Long,
    private val channels: Long,
    private val imageTransform: ImageTransform?,
    private val files: Array<File>,
    private val alwaysReturningSameResult: Boolean = false,
    val testBatches: Array<DataSet?>?
) : BaseRecordReader() {
    private var iter: Iterator<File>
    private var currentFile: File? = null
    private var labels = listOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    private var imageLoader: NativeImageLoader? = null
    private lateinit var conf: Configuration
    private var sameResultToReturn: List<List<Writable?>?>? = null
    private var alwaysReturningSameResultDone = false

    init {
        if (imageLoader == null) {
            imageLoader = NativeImageLoader(height, width, channels, imageTransform)
        }
        iter = files.iterator()
    }


    override fun initialize(split: InputSplit) {
        throw NotImplementedError("Don't use this function")
    }


    override fun initialize(conf: Configuration, split: InputSplit) {
        throw NotImplementedError("Don't use this function")
    }


    override fun next(): List<Writable> {
        val ret: MutableList<Writable>
        val image = iter.next()
        currentFile = image
        if (image.isDirectory) return next()
        try {
            invokeListeners(image)
            val row = imageLoader!!.asMatrix(image)
            Nd4j.getAffinityManager().ensureLocation(row, AffinityManager.Location.DEVICE)
            ret = RecordConverter.toRecord(row)
            ret.add(IntWritable(labels.indexOf(getLabel(image.path))))
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException(e)
        }
        return ret
    }

    override fun hasNext(): Boolean {
        return if (alwaysReturningSameResultDone) false
        else iter.hasNext()
    }

    override fun getLabels() = labels

    override fun batchesSupported() = imageLoader is NativeImageLoader

    override fun next(num: Int): List<List<Writable?>?> {
        if (alwaysReturningSameResult && sameResultToReturn != null) {
            alwaysReturningSameResultDone = true
            return sameResultToReturn!!
        }
        Preconditions.checkArgument(num > 0, "Number of examples must be > 0: got $num")
        if (imageLoader == null) {
            imageLoader = NativeImageLoader(height, width, channels, imageTransform)
        }
        val currBatch: MutableList<File> = ArrayList()
        var cnt = 0
        val numCategories = labels.size
        var currLabels: MutableList<Int?>? = null
        while (cnt < num && iter.hasNext()) {
            currentFile = iter.next()
            currBatch.add(currentFile!!)
            invokeListeners(currentFile)
            //Collect the label Writables from the label generators
            if (currLabels == null) currLabels = ArrayList()
            currLabels.add(labels.indexOf(getLabel(currentFile!!.path)))
            cnt++
        }
        val features = Nd4j.createUninitialized(longArrayOf(cnt.toLong(), channels, height, width), 'c')
        Nd4j.getAffinityManager().tagLocation(features, AffinityManager.Location.HOST)
        for (i in 0 until cnt) {
            try {
                imageLoader!!.asMatrixView(
                    currBatch[i],
                    features.tensorAlongDimension(i.toLong(), 1, 2, 3)
                )
            } catch (e: Exception) {
                println("Image file failed during load: " + currBatch[i].absolutePath)
                throw RuntimeException(e)
            }
        }
        Nd4j.getAffinityManager().ensureLocation(features, AffinityManager.Location.DEVICE)
        val ret: MutableList<INDArray> = ArrayList()
        ret.add(features)
        //And convert the previously collected label Writables from the label generators
        //Standard classification use case (i.e., handle String -> integer conversion)
        val labels = Nd4j.create(cnt.toLong(), numCategories.toLong(), 'c')
        Nd4j.getAffinityManager().tagLocation(labels, AffinityManager.Location.HOST)
        for (i in currLabels!!.indices) {
            labels.putScalar(i.toLong(), currLabels[i]!!.toLong(), 1.0)
        }
        ret.add(labels)
        val res = NDArrayRecordBatch(ret)
        if (alwaysReturningSameResult) sameResultToReturn = res
        return res
    }

    @Throws(IOException::class)
    override fun close() {
        //No op
    }

    override fun setConf(conf: Configuration) {
        this.conf = conf
    }

    override fun getConf(): Configuration {
        return conf
    }

    /**
     * Get the label from the given path
     *
     * @param path the path to get the label from
     * @return the label for the given path
     */
    private fun getLabel(path: String): String {
        val split = path.split('/')
        return split[split.size - 2]
    }

    override fun reset() {
        iter = files.iterator()
        alwaysReturningSameResultDone = false
    }

    override fun resetSupported() = true

    override fun record(uri: URI, dataInputStream: DataInputStream?): List<Writable>? {
        invokeListeners(uri)
        if (imageLoader == null) {
            imageLoader = NativeImageLoader(height, width, channels, imageTransform)
        }
        val row = imageLoader!!.asMatrix(dataInputStream)
        val ret = RecordConverter.toRecord(row)
        ret.add(IntWritable(labels.indexOf(getLabel(uri.path))))
        return ret
    }

    override fun nextRecord(): Record {
        val list = next()
        val uri = URIUtil.fileToURI(currentFile)
        return org.datavec.api.records.impl.Record(list, RecordMetaDataURI(uri, BaseImageRecordReader::class.java))
    }

    override fun loadFromMetaData(recordMetaData: RecordMetaData): Record {
        return loadFromMetaData(listOf(recordMetaData))[0]
    }

    override fun loadFromMetaData(recordMetaDatas: List<RecordMetaData>): List<Record> {
        val out: MutableList<Record> = ArrayList()
        for (meta in recordMetaDatas) {
            val uri = meta.uri
            val f = File(uri)
            var next: List<Writable>?
            DataInputStream(BufferedInputStream(FileInputStream(f))).use { dis -> next = record(uri, dis) }
            out.add(org.datavec.api.records.impl.Record(next, meta))
        }
        return out
    }
}
