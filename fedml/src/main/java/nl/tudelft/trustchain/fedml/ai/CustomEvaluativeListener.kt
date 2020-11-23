package nl.tudelft.trustchain.fedml.ai

import mu.KotlinLogging
import org.deeplearning4j.exception.DL4JInvalidInputException
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.InvocationType
import org.nd4j.evaluation.IEvaluation
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import java.util.concurrent.atomic.AtomicLong


private val logger = KotlinLogging.logger("CustomEvaluativeListener")

class CustomEvaluativeListener {
    @Transient
    private var iterationCount = ThreadLocal<AtomicLong?>()
    private var frequency: Int
    private var invocationCount = AtomicLong(0)

    @Transient
    private var dsIterator: DataSetIterator? = null

    @Transient
    private var mdsIterator: MultiDataSetIterator? = null
    private var ds: DataSet? = null
    private var mds: MultiDataSet? = null

    private var evaluations: Array<out IEvaluation<*>>

    private var invocationType: InvocationType

    @Transient
    var callback: CustomEvaluationCallback? = null

    /**
     * Evaluation will be launched after each *frequency* iterations, with [Evaluation] datatype
     * @param iterator  Iterator to provide data for evaluation
     * @param frequency Frequency (in number of iterations) to perform evaluation
     */
    constructor(iterator: DataSetIterator, frequency: Int) : this(
        iterator,
        frequency,
        InvocationType.ITERATION_END,
        Evaluation()
    )

    /**
     * Evaluation will be launched after each *frequency* iteration
     *
     * @param iterator    Iterator to provide data for evaluation
     * @param frequency   Frequency (in number of iterations/epochs according to the invocation type) to perform evaluation
     * @param type        Type of value for 'frequency' - iteration end, epoch end, etc
     * @param evaluations Type of evalutions to perform
     */
    constructor(
        iterator: DataSetIterator, frequency: Int, type: InvocationType,
        vararg evaluations: IEvaluation<*>
    ) {
        dsIterator = iterator
        this.frequency = frequency
        this.evaluations = evaluations
        invocationType = type
    }

    fun iterationDone(model: Model, logging: Boolean) {
        if (invocationType == InvocationType.ITERATION_END) invokeListener(model, logging)
    }

    private fun invokeListener(model: Model, logging: Boolean) {
        if (iterationCount.get() == null) iterationCount.set(AtomicLong(0))
        if (iterationCount.get()!!.getAndIncrement() % frequency != 0L) return
        for (evaluation in evaluations) evaluation.reset()
        if (dsIterator != null && dsIterator!!.resetSupported()) dsIterator!!.reset() else if (mdsIterator != null && mdsIterator!!.resetSupported()) mdsIterator!!.reset()

        if (logging) logger.debug { "Starting evaluation nr. ${invocationCount.incrementAndGet()}" }
        if (model is MultiLayerNetwork) {
            if (dsIterator != null) {
                model.doEvaluation(dsIterator, *evaluations)
            } else if (ds != null) {
                for (evaluation in evaluations) evaluation.eval(
                    ds!!.labels, model.output(
                        ds!!.features
                    )
                )
            }
        } else if (model is ComputationGraph) {
            if (dsIterator != null) {
                model.doEvaluation(dsIterator, *evaluations)
            } else if (mdsIterator != null) {
                model.doEvaluation(mdsIterator, *evaluations)
            } else if (ds != null) {
                for (evaluation in evaluations) evalAtIndex(
                    evaluation, arrayOf(
                        ds!!.labels
                    ),
                    model.output(ds!!.features), 0
                )
            } else if (mds != null) {
                for (evaluation in evaluations) evalAtIndex(evaluation, mds!!.labels, model.output(*mds!!.features), 0)
            }
        } else throw DL4JInvalidInputException("Model is unknown: " + model.javaClass.canonicalName)

        if (logging) logger.debug { "Reporting evaluation results:" }
        for (evaluation in evaluations) {
            if (logging) logger.debug { "${evaluation.javaClass.simpleName}:\n${evaluation.stats()}" }
        }
        if (callback != null) callback!!.call(this, model, invocationCount.get(), evaluations)
    }

    private fun evalAtIndex(evaluation: IEvaluation<*>, labels: Array<INDArray?>, predictions: Array<INDArray?>, index: Int) {
        evaluation.eval(labels[index], predictions[index])
    }
}
