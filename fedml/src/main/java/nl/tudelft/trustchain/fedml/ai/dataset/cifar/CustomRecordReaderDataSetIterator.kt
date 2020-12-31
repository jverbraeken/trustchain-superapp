package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import org.datavec.api.io.WritableConverter
import org.datavec.api.io.converters.SelfWritableConverter
import org.datavec.api.records.Record
import org.datavec.api.records.metadata.RecordMetaDataComposableMap
import org.datavec.api.writable.Writable
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
import org.nd4j.common.base.Preconditions
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable
import java.util.*

open class CustomRecordReaderDataSetIterator : DataSetIterator {
    protected var recordReader: CustomImageRecordReader
    private var converter: WritableConverter?
    private var batchSize = 10
    private var maxNumBatches = -1
    private var batchNum = 0
    private var labelIndex = -1
    private var labelIndexTo = -1
    private var numPossibleLabels = -1
    private var sequenceIter: Iterator<List<Writable>>? = null
    private var last: DataSet? = null
    private var useCurrent = false
    private var regression = false

    private var preProcessor: DataSetPreProcessor? = null

    private var collectMetaData = false
    private var underlying: RecordReaderMultiDataSetIterator? = null
    private var underlyingIsDisjoint = false

    constructor(
        recordReader: CustomImageRecordReader, batchSize: Int, labelIndex: Int,
        numPossibleLabels: Int,
    ) : this(recordReader, SelfWritableConverter(), batchSize, labelIndex, labelIndex, numPossibleLabels, -1, false)

    constructor(
        recordReader: CustomImageRecordReader, converter: WritableConverter?, batchSize: Int,
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
        val builder = RecordReaderMultiDataSetIterator.Builder(batchSize)
        builder.addReader(READER_KEY, recordReader)
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
            when {
                labelIndex < 0 -> {
                    //No label
                    inputFrom = 0
                    inputTo = totalSize - 1
                }
                labelIndex == 0 -> {
                    inputFrom = labelIndexTo + 1
                    inputTo = totalSize - 1
                }
                else -> {
                    inputFrom = 0
                    inputTo = labelIndex - 1
                }
            }
            builder.addInput(READER_KEY, inputFrom, inputTo)
            underlyingIsDisjoint = false
        } else if (labelIndex >= 0) {
            Preconditions.checkState(
                labelIndex < next.record.size,
                "Invalid label (from) index: index must be in range 0 to first record size of (0 to %s inclusive), got %s",
                next.record.size - 1,
                labelIndex
            )
            Preconditions.checkState(
                labelIndexTo < next.record.size,
                "Invalid label (to) index: index must be in range 0 to first record size of (0 to %s inclusive), got %s",
                next.record.size - 1,
                labelIndexTo
            )


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
            ds.exampleMetaData = temp2
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
        return recordReader.labels.toList()
    }

    companion object {
        private const val READER_KEY = "reader"

        //Package private
        fun getOrNull(arr: Array<INDArray?>?, idx: Int): INDArray? {
            return if (arr == null || arr.isEmpty()) {
                null
            } else arr[idx]
        }
    }
}
