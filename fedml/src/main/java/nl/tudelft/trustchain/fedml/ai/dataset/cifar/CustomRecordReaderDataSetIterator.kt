package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import org.datavec.api.io.WritableConverter
import org.datavec.api.io.converters.SelfWritableConverter
import org.datavec.api.records.Record
import org.datavec.api.records.metadata.RecordMetaData
import org.datavec.api.records.metadata.RecordMetaDataComposableMap
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.SequenceRecordReader
import org.datavec.api.records.reader.impl.ConcatenatingRecordReader
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.datavec.api.writable.Writable
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
import org.nd4j.base.Preconditions
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.io.IOException
import java.io.Serializable
import java.util.*

open class CustomRecordReaderDataSetIterator : DataSetIterator {
    protected var recordReader: RecordReader
    protected var converter: WritableConverter?
    protected var batchSize = 10
    protected var maxNumBatches = -1
    protected var batchNum = 0
    protected var labelIndex = -1
    protected var labelIndexTo = -1
    protected var numPossibleLabels = -1
    protected var sequenceIter: Iterator<List<Writable>>? = null
    protected var last: DataSet? = null
    protected var useCurrent = false
    protected var regression = false

    private var preProcessor: DataSetPreProcessor? = null

    private var collectMetaData = false
    private var underlying: RecordReaderMultiDataSetIterator? = null
    private var underlyingIsDisjoint = false

    /**
     * Constructor for classification, where:<br></br>
     * (a) the label index is assumed to be the very last Writable/column, and<br></br>
     * (b) the number of classes is inferred from RecordReader.getLabels()<br></br>
     * Note that if RecordReader.getLabels() returns null, no output labels will be produced
     *
     * @param recordReader Record reader to use as the source of data
     * @param batchSize    Minibatch size, for each call of .next()
     */
    constructor(recordReader: RecordReader, batchSize: Int) : this(recordReader, SelfWritableConverter(), batchSize, -1, -1,
        if (recordReader.labels == null) -1 else recordReader.labels.size, -1, false) {
    }

    /**
     * Main constructor for classification. This will convert the input class index (at position labelIndex, with integer
     * values 0 to numPossibleLabels-1 inclusive) to the appropriate one-hot output/labels representation.
     *
     * @param recordReader         RecordReader: provides the source of the data
     * @param batchSize            Batch size (number of examples) for the output DataSet objects
     * @param labelIndex           Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
     * @param numPossibleLabels    Number of classes (possible labels) for classification
     */
    constructor(
        recordReader: RecordReader, batchSize: Int, labelIndex: Int,
        numPossibleLabels: Int,
    ) : this(recordReader, SelfWritableConverter(), batchSize, labelIndex, labelIndex, numPossibleLabels, -1, false) {
    }

    /**
     * Constructor for classification, where the maximum number of returned batches is limited to the specified value
     *
     * @param recordReader      the recordreader to use
     * @param labelIndex        the index/column of the label (for classification)
     * @param numPossibleLabels the number of possible labels for classification. Not used if regression == true
     * @param maxNumBatches     The maximum number of batches to return between resets. Set to -1 to return all available data
     */
    constructor(
        recordReader: RecordReader, batchSize: Int, labelIndex: Int, numPossibleLabels: Int,
        maxNumBatches: Int,
    ) : this(recordReader, SelfWritableConverter(), batchSize, labelIndex, labelIndex, numPossibleLabels, maxNumBatches, false) {
    }

    /**
     * Main constructor for multi-label regression (i.e., regression with multiple outputs). Can also be used for single
     * output regression with labelIndexFrom == labelIndexTo
     *
     * @param recordReader      RecordReader to get data from
     * @param labelIndexFrom    Index of the first regression target
     * @param labelIndexTo      Index of the last regression target, inclusive
     * @param batchSize         Minibatch size
     * @param regression        Require regression = true. Mainly included to avoid clashing with other constructors previously defined :/
     */
    constructor(
        recordReader: RecordReader, batchSize: Int, labelIndexFrom: Int, labelIndexTo: Int,
        regression: Boolean,
    ) : this(recordReader, SelfWritableConverter(), batchSize, labelIndexFrom, labelIndexTo, -1, -1, regression) {
        require(regression) {
            "This constructor is only for creating regression iterators. " +
                "If you're doing classification you need to use another constructor that " +
                "(implicitly) specifies numPossibleLabels"
        }
    }

    /**
     * Main constructor
     *
     * @param recordReader      the recordreader to use
     * @param converter         Converter. May be null.
     * @param batchSize         Minibatch size - number of examples returned for each call of .next()
     * @param labelIndexFrom    the index of the label (for classification), or the first index of the labels for multi-output regression
     * @param labelIndexTo      only used if regression == true. The last index *inclusive* of the multi-output regression
     * @param numPossibleLabels the number of possible labels for classification. Not used if regression == true
     * @param maxNumBatches     Maximum number of batches to return
     * @param regression        if true: regression. If false: classification (assume labelIndexFrom is the class it belongs to)
     */
    constructor(
        recordReader: RecordReader, converter: WritableConverter?, batchSize: Int,
        labelIndexFrom: Int, labelIndexTo: Int, numPossibleLabels: Int, maxNumBatches: Int,
        regression: Boolean,
    ) {
        this.recordReader = recordReader
        this.converter = converter
        this.batchSize = batchSize
        this.maxNumBatches = maxNumBatches
        labelIndex = labelIndexFrom
        this.labelIndexTo = labelIndexTo
        this.numPossibleLabels = numPossibleLabels
        this.regression = regression
    }

    protected constructor(b: Builder) {
        recordReader = b.recordReader
        converter = b.converter
        batchSize = b.batchSize
        maxNumBatches = b.maxNumBatches
        labelIndex = b.labelIndex
        labelIndexTo = b.labelIndexTo
        numPossibleLabels = b.numPossibleLabels
        regression = b.regression
        preProcessor = b.preProcessor
    }

    /**
     * When set to true: metadata for  the current examples will be present in the returned DataSet.
     * Disabled by default.
     *
     * @param collectMetaData Whether to collect metadata or  not
     */
    fun setCollectMetaData(collectMetaData: Boolean) {
        if (underlying != null) {
            underlying!!.isCollectMetaData = collectMetaData
        }
        this.collectMetaData = collectMetaData
    }

    private fun initializeUnderlying() {
        if (underlying == null) {
            val next = recordReader.nextRecord()
            initializeUnderlying(next)
        }
    }

    private fun initializeUnderlying(next: Record) {
        val totalSize = next.record.size

        //allow people to specify label index as -1 and infer the last possible label
        if (numPossibleLabels >= 1 && labelIndex < 0) {
            labelIndex = totalSize - 1
            labelIndexTo = labelIndex
        }
        if (recordReader.resetSupported()) {
            recordReader.reset()
        } else {
            //Hack around the fact that we need the first record to initialize the underlying RRMDSI, but can't reset
            // the original reader
            recordReader = ConcatenatingRecordReader(
                CollectionRecordReader(listOf(next.record)),
                recordReader)
        }
        val builder = RecordReaderMultiDataSetIterator.Builder(batchSize)
        if (recordReader is SequenceRecordReader) {
            builder.addSequenceReader(READER_KEY, recordReader as SequenceRecordReader)
        } else {
            builder.addReader(READER_KEY, recordReader)
        }
        if (regression) {
            builder.addOutput(READER_KEY, labelIndex, labelIndexTo)
        } else if (numPossibleLabels >= 1) {
            builder.addOutputOneHot(READER_KEY, labelIndex, numPossibleLabels)
        }

        //Inputs: assume to be all of the other writables
        //In general: can't assume label indices are all at the start or end (event though 99% of the time they are)
        //If they are: easy. If not: use 2 inputs in the underlying as a workaround, and concat them
        if (labelIndex >= 0 && (labelIndex == 0 || labelIndexTo == totalSize - 1)) {
            //Labels are first or last -> one input in underlying
            val inputFrom: Int
            val inputTo: Int
            if (labelIndex < 0) {
                //No label
                inputFrom = 0
                inputTo = totalSize - 1
            } else if (labelIndex == 0) {
                inputFrom = labelIndexTo + 1
                inputTo = totalSize - 1
            } else {
                inputFrom = 0
                inputTo = labelIndex - 1
            }
            builder.addInput(READER_KEY, inputFrom, inputTo)
            underlyingIsDisjoint = false
        } else if (labelIndex >= 0) {
            Preconditions.checkState(labelIndex < next.record.size,
                "Invalid label (from) index: index must be in range 0 to first record size of (0 to %s inclusive), got %s",
                next.record.size - 1,
                labelIndex)
            Preconditions.checkState(labelIndexTo < next.record.size,
                "Invalid label (to) index: index must be in range 0 to first record size of (0 to %s inclusive), got %s",
                next.record.size - 1,
                labelIndexTo)


            //Multiple inputs
            val firstFrom = 0
            val firstTo = labelIndex - 1
            val secondFrom = labelIndexTo + 1
            val secondTo = totalSize - 1
            builder.addInput(READER_KEY, firstFrom, firstTo)
            builder.addInput(READER_KEY, secondFrom, secondTo)
            underlyingIsDisjoint = true
        } else {
            //No labels - only features
            builder.addInput(READER_KEY)
            underlyingIsDisjoint = false
        }
        underlying = builder.build()
        if (collectMetaData) {
            underlying!!.isCollectMetaData = true
        }
    }

    private fun mdsToDataSet(mds: MultiDataSet): DataSet {
        val f: INDArray?
        val fm: INDArray?
        if (underlyingIsDisjoint) {
            //Rare case: 2 input arrays -> concat
            val f1 = getOrNull(mds.features, 0)
            val f2 = getOrNull(mds.features, 1)
            fm = getOrNull(mds.featuresMaskArrays, 0) //Per-example masking only on the input -> same for both

            //Can assume 2d features here
            f = Nd4j.hstack(f1, f2)
        } else {
            //Standard case
            f = getOrNull(mds.features, 0)
            fm = getOrNull(mds.featuresMaskArrays, 0)
        }
        val l = getOrNull(mds.labels, 0)
        val lm = getOrNull(mds.labelsMaskArrays, 0)
        val ds = DataSet(f, l, fm, lm)
        if (collectMetaData) {
            val temp = mds.exampleMetaData
            val temp2: MutableList<Serializable?> = ArrayList(temp.size)
            for (s in temp) {
                val m = s as RecordMetaDataComposableMap
                temp2.add(m.meta[READER_KEY])
            }
            ds.setExampleMetaData(temp2)
        }

        //Edge case, for backward compatibility:
        //If labelIdx == -1 && numPossibleLabels == -1 -> no labels -> set labels array to features array
        if (labelIndex == -1 && numPossibleLabels == -1 && ds.labels == null) {
            ds.labels = ds.features
        }
        if (preProcessor != null) {
            preProcessor!!.preProcess(ds)
        }
        return ds
    }

    override fun next(num: Int): DataSet {
        if (useCurrent) {
            useCurrent = false
            if (preProcessor != null) preProcessor!!.preProcess(last)
            return last!!
        }
        if (underlying == null) {
            initializeUnderlying()
        }
        batchNum++
        return mdsToDataSet(underlying!!.next(num))
    }

    override fun inputColumns(): Int {
        return if (last == null) {
            val next = next()
            last = next
            useCurrent = true
            next.numInputs()
        } else last!!.numInputs()
    }

    override fun totalOutcomes(): Int {
        return if (last == null) {
            val next = next()
            last = next
            useCurrent = true
            next.numOutcomes()
        } else last!!.numOutcomes()
    }

    override fun resetSupported(): Boolean {
        if (underlying == null) {
            initializeUnderlying()
        }
        return underlying!!.resetSupported()
    }

    override fun asyncSupported(): Boolean {
        return true
    }

    override fun reset() {
        batchNum = 0
        if (underlying != null) {
            underlying!!.reset()
        }
        last = null
        useCurrent = false
    }

    override fun batch(): Int {
        return batchSize
    }

    override fun setPreProcessor(preProcessor: DataSetPreProcessor) {
        this.preProcessor = preProcessor
    }

    override fun getPreProcessor(): DataSetPreProcessor? {
        return preProcessor
    }

    override fun hasNext(): Boolean {
        return ((sequenceIter != null && sequenceIter!!.hasNext() || recordReader.hasNext())
            && (maxNumBatches < 0 || batchNum < maxNumBatches))
    }

    override fun next(): DataSet {
        return next(batchSize)
    }

    override fun remove() {
        throw UnsupportedOperationException()
    }

    override fun getLabels(): List<String> {
        return recordReader.labels
    }

    /**
     * Load a single example to a DataSet, using the provided RecordMetaData.
     * Note that it is more efficient to load multiple instances at once, using [.loadFromMetaData]
     *
     * @param recordMetaData RecordMetaData to load from. Should have been produced by the given record reader
     * @return DataSet with the specified example
     * @throws IOException If an error occurs during loading of the data
     */
    @Throws(IOException::class)
    fun loadFromMetaData(recordMetaData: RecordMetaData?): DataSet {
        return loadFromMetaData(listOf(recordMetaData))
    }

    /**
     * Load a multiple examples to a DataSet, using the provided RecordMetaData instances.
     *
     * @param list List of RecordMetaData instances to load from. Should have been produced by the record reader provided
     * to the RecordReaderDataSetIterator constructor
     * @return DataSet with the specified examples
     * @throws IOException If an error occurs during loading of the data
     */
    @Throws(IOException::class)
    fun loadFromMetaData(list: List<RecordMetaData?>): DataSet {
        if (underlying == null) {
            val r = recordReader.loadFromMetaData(list[0])
            initializeUnderlying(r)
        }

        //Convert back to composable:
        val l: MutableList<RecordMetaData> = ArrayList(list.size)
        for (m in list) {
            l.add(RecordMetaDataComposableMap(Collections.singletonMap(READER_KEY, m)))
        }
        val m = underlying!!.loadFromMetaData(l)
        return mdsToDataSet(m)
    }

    class Builder(var recordReader: RecordReader, var batchSize: Int) {
        var converter: WritableConverter? = null
        var maxNumBatches = -1
        var labelIndex = -1
        var labelIndexTo = -1
        var numPossibleLabels = -1
        var regression = false
        var preProcessor: DataSetPreProcessor? = null
        private var collectMetaData = false
        private var clOrRegCalled = false
        fun writableConverter(converter: WritableConverter?): Builder {
            this.converter = converter
            return this
        }

        /**
         * Optional argument, usually not used. If set, can be used to limit the maximum number of minibatches that
         * will be returned (between resets). If not set, will always return as many minibatches as there is data
         * available.
         *
         * @param maxNumBatches Maximum number of minibatches per epoch / reset
         */
        fun maxNumBatches(maxNumBatches: Int): Builder {
            this.maxNumBatches = maxNumBatches
            return this
        }

        /**
         * Use this for single output regression (i.e., 1 output/regression target)
         *
         * @param labelIndex Column index that contains the regression target (indexes start at 0)
         */
        fun regression(labelIndex: Int): Builder {
            return regression(labelIndex, labelIndex)
        }

        /**
         * Use this for multiple output regression (1 or more output/regression targets). Note that all regression
         * targets must be contiguous (i.e., positions x to y, without gaps)
         *
         * @param labelIndexFrom Column index of the first regression target (indexes start at 0)
         * @param labelIndexTo   Column index of the last regression target (inclusive)
         */
        fun regression(labelIndexFrom: Int, labelIndexTo: Int): Builder {
            labelIndex = labelIndexFrom
            this.labelIndexTo = labelIndexTo
            regression = true
            clOrRegCalled = true
            return this
        }

        /**
         * Use this for classification
         *
         * @param labelIndex Index that contains the label index. Column (indexes start from 0) be an integer value,
         * and contain values 0 to numClasses-1
         * @param numClasses Number of label classes (i.e., number of categories/classes in the dataset)
         */
        fun classification(labelIndex: Int, numClasses: Int): Builder {
            this.labelIndex = labelIndex
            labelIndexTo = labelIndex
            numPossibleLabels = numClasses
            regression = false
            clOrRegCalled = true
            return this
        }

        /**
         * Optional arg. Allows the preprocessor to be set
         * @param preProcessor Preprocessor to use
         */
        fun preProcessor(preProcessor: DataSetPreProcessor?): Builder {
            this.preProcessor = preProcessor
            return this
        }

        /**
         * When set to true: metadata for  the current examples will be present in the returned DataSet.
         * Disabled by default.
         *
         * @param collectMetaData Whether metadata should be collected or not
         */
        fun collectMetaData(collectMetaData: Boolean): Builder {
            this.collectMetaData = collectMetaData
            return this
        }

        fun build(): CustomRecordReaderDataSetIterator {
            return CustomRecordReaderDataSetIterator(this)
        }
    }

    companion object {
        private const val READER_KEY = "reader"

        //Package private
        fun getOrNull(arr: Array<INDArray?>?, idx: Int): INDArray? {
            return if (arr == null || arr.size == 0) {
                null
            } else arr[idx]
        }
    }
}
