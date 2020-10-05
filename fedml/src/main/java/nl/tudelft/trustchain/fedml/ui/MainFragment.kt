package nl.tudelft.trustchain.fedml.ui

import android.os.Bundle
import android.os.StrictMode
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import nl.tudelft.trustchain.common.ui.BaseFragment
import nl.tudelft.trustchain.common.util.viewBinding
import nl.tudelft.trustchain.fedml.R
import nl.tudelft.trustchain.fedml.ai.*
import nl.tudelft.trustchain.fedml.databinding.FragmentMainBinding
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import nl.tudelft.trustchain.fedml.ipv8.MsgPing
import org.deeplearning4j.common.resources.DL4JResources
import java.io.File

class MainFragment : BaseFragment(R.layout.fragment_main), AdapterView.OnItemSelectedListener {
    private val baseDirectory: File by lazy { requireActivity().filesDir }
    private val binding by viewBinding(FragmentMainBinding::bind)

    private val datasets = Datasets.values().map { it.identifier }
    private val updaters = Updaters.values().map { it.identifier }
    private val learningRates = LearningRates.values().map { it.identifier }
    private val momentums = Momentums.values().map { it.identifier }
    private val l2Regularizations = L2Regularizations.values().map { it.identifier }
    private val batchSizes = BatchSizes.values().map { it.identifier }

    private var dataset: Datasets = Datasets.MNIST
    private var updater: Updaters = Datasets.MNIST.defaultUpdater
    private var learningRate: LearningRates = Datasets.MNIST.defaultLearningRate
    private var momentum: Momentums? = null
    private var l2: L2Regularizations = Datasets.MNIST.defaultL2
    private var batchSize: BatchSizes = Datasets.MNIST.defaultBatchSize

    private fun getCommunity(): FedMLCommunity {
        return getIpv8().getOverlay()
            ?: throw java.lang.IllegalStateException("FedMLCommunity is not configured")
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        bindSpinner(view, binding.spnDataset, datasets)
        bindSpinner(view, binding.spnUpdater, updaters)
        bindSpinner(view, binding.spnLearningRate, learningRates)
        bindSpinner(view, binding.spnMomentum, momentums)
        bindSpinner(view, binding.spnL2Regularization, l2Regularizations)
        bindSpinner(view, binding.spnBatchSize, batchSizes)

        binding.btnPing.setOnClickListener { onBtnPingClicked() }
        binding.btnRunLocal.setOnClickListener { onBtnRunLocallyClicked() }
        binding.btnRunDistrSim.setOnClickListener { onBtnSimulateDistributedLocallyClicked() }
        binding.btnRunDistr.setOnClickListener { onBtnRunDistributedClicked() }

        configureDL4JDirectory()
        allowDL4JOnUIThread()
        binding.spnDataset.setSelection(datasets.indexOf(dataset.identifier))
        setSpinnersToDataset(dataset)
    }

    private fun bindSpinner(view: View, spinner: Spinner, elements: List<String>) {
        spinner.onItemSelectedListener = this
        val adapter = ArrayAdapter(view.context, android.R.layout.simple_spinner_item, elements)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinner.adapter = adapter
    }

    private fun configureDL4JDirectory() {
        DL4JResources.setBaseDirectory(baseDirectory)
    }

    private fun allowDL4JOnUIThread() {
        val policy: StrictMode.ThreadPolicy = StrictMode.ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)
    }

    ////// BUTTON CLICK LISTENERS

    private fun onBtnPingClicked() {
        getCommunity().sendToAll(MessageId.MSG_PING, MsgPing("Ping"))
    }

    private fun onBtnRunLocallyClicked() {
        LocalRunner().run(
            baseDirectory,
            dataset,
            updater,
            learningRate,
            momentum,
            l2,
            batchSize
        )
    }

    private fun onBtnSimulateDistributedLocallyClicked() {
        SimulatedRunner().run(
            baseDirectory,
            dataset,
            updater,
            learningRate,
            momentum,
            l2,
            batchSize
        )
    }

    private fun onBtnRunDistributedClicked() {
        DistributedRunner(getCommunity()).run(
            baseDirectory,
            dataset,
            updater,
            learningRate,
            momentum,
            l2,
            batchSize
        )
    }

    override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
        when (parent!!.id) {
            binding.spnDataset.id -> {
                dataset = Datasets.values().first { it.identifier == datasets[position] }
                setSpinnersToDataset(dataset)
            }
            binding.spnUpdater.id -> updater =
                Updaters.values().first { it.identifier == updaters[position] }
            binding.spnLearningRate.id -> learningRate =
                LearningRates.values().first { it.identifier == learningRates[position] }
            binding.spnMomentum.id -> momentum =
                Momentums.values().first { it.identifier == momentums[position] }
            binding.spnL2Regularization.id -> l2 =
                L2Regularizations.values().first { it.identifier == l2Regularizations[position] }
            binding.spnBatchSize.id -> batchSize =
                BatchSizes.values().first { it.identifier == batchSizes[position] }
        }
    }

    private fun setSpinnersToDataset(dataset: Datasets) {
        binding.spnUpdater.setSelection(updaters.indexOf(dataset.defaultUpdater.identifier), true)
        binding.spnLearningRate.setSelection(
            learningRates.indexOf(dataset.defaultLearningRate.identifier),
            true
        )
        binding.spnMomentum.setSelection(
            momentums.indexOf(dataset.defaultMomentum.identifier),
            true
        )
        binding.spnL2Regularization.setSelection(
            l2Regularizations.indexOf(dataset.defaultL2.identifier),
            true
        )
        binding.spnBatchSize.setSelection(
            batchSizes.indexOf(dataset.defaultBatchSize.identifier),
            true
        )
    }

    override fun onNothingSelected(parent: AdapterView<*>?) {
        // Nothing
    }
}
