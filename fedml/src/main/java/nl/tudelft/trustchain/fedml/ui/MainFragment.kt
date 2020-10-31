package nl.tudelft.trustchain.fedml.ui

import android.content.res.AssetManager
import android.os.Bundle
import android.os.StrictMode
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import mu.KotlinLogging
import nl.tudelft.trustchain.common.ui.BaseFragment
import nl.tudelft.trustchain.common.util.viewBinding
import nl.tudelft.trustchain.fedml.R
import nl.tudelft.trustchain.fedml.ai.*
import nl.tudelft.trustchain.fedml.databinding.FragmentMainBinding
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import nl.tudelft.trustchain.fedml.ipv8.MsgPing
import org.deeplearning4j.common.resources.DL4JResources
import java.io.*

private val logger = KotlinLogging.logger("FedML.MainFragment")

//-e activity fedml -e dataset mnist -e optimizer adam -e learningRate rate_1em3 -e momentum none -e l2Regularization l2_5em3 -e batchSize batch_32 -e epoch epoch_50 -e runner distributed -e run true
//-e activity fedml -e dataset cifar10 -e optimizer sgd -e learningRate schedule1 -e momentum momentum_1em3 -e l2Regularization l2_1em4 -e batchSize batch_5 -e epoch epoch_25 -e runner distributed -e run true
class MainFragment : BaseFragment(R.layout.fragment_main), AdapterView.OnItemSelectedListener {
    private val baseDirectory: File by lazy { requireActivity().filesDir }
    private val binding by viewBinding(FragmentMainBinding::bind)
    private val seed = getCommunity().myEstimatedLan.toString().hashCode()

    private val datasets = Datasets.values().map { it.text }
    private val optimizers = Optimizers.values().map { it.text }
    private val learningRates = LearningRates.values().map { it.text }
    private val momentums = Momentums.values().map { it.text }
    private val l2Regularizations = L2Regularizations.values().map { it.text }
    private val batchSizes = BatchSizes.values().map { it.text }
    private val epochs = Epochs.values().map { it.text }
    private val iteratorDistributions = IteratorDistributions.values().map { it.text }
    private val maxTestSamples = MaxTestSamples.values().map { it.text }

    private var dataset: Datasets = Datasets.MNIST
    private var optimizer: Optimizers = dataset.defaultOptimizer
    private var learningRate: LearningRates = dataset.defaultLearningRate
    private var momentum: Momentums = dataset.defaultMomentum
    private var l2: L2Regularizations = dataset.defaultL2
    private var batchSize: BatchSizes = dataset.defaultBatchSize
    private var epoch: Epochs = Epochs.EPOCH_5
    private var iteratorDistribution: IteratorDistributions = dataset.defaultIteratorDistribution
    private var maxTestSample = MaxTestSamples.NUM_50

    private fun getCommunity(): FedMLCommunity {
        return getIpv8().getOverlay()
            ?: throw java.lang.IllegalStateException("FedMLCommunity is not configured")
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        bindSpinner(view, binding.spnDataset, datasets)
        bindSpinner(view, binding.spnOptimizer, optimizers)
        bindSpinner(view, binding.spnLearningRate, learningRates)
        bindSpinner(view, binding.spnMomentum, momentums)
        bindSpinner(view, binding.spnL2Regularization, l2Regularizations)
        bindSpinner(view, binding.spnBatchSize, batchSizes)
        bindSpinner(view, binding.spnEpochs, epochs)
        bindSpinner(view, binding.spnIteratorDistribution, iteratorDistributions)
        bindSpinner(view, binding.spnMaxTestSamples, maxTestSamples)

        binding.btnPing.setOnClickListener { onBtnPingClicked() }
        binding.btnRunLocal.setOnClickListener { onBtnRunLocallyClicked() }
        binding.btnRunDistrSim.setOnClickListener { onBtnSimulateDistributedLocallyClicked() }
        binding.btnRunDistr.setOnClickListener { onBtnRunDistributedClicked() }

        configureDL4JDirectory()
        allowDL4JOnUIThread()
        binding.spnDataset.setSelection(datasets.indexOf(dataset.text))
        binding.spnEpochs.setSelection(datasets.indexOf(epoch.text))
        binding.spnMaxTestSamples.setSelection(datasets.indexOf(maxTestSample.text))
        processIntentExtras()
        synchronizeSpinners()

        copyAssets()

        if (requireActivity().intent?.extras?.getString("run") == "true") {
            when (requireActivity().intent?.extras?.getString("runner")) {
                "local" -> onBtnRunLocallyClicked()
                "simulated" -> onBtnSimulateDistributedLocallyClicked()
                "distributed" -> onBtnRunDistributedClicked()
                else -> throw IllegalStateException("Runner must be either local, simulated, or distributed")
            }
        }
    }

    private fun processIntentExtras() {
        val extras = requireActivity().intent?.extras

        val dataset = extras?.getString("dataset")
        if (dataset != null) {
            this.dataset = Datasets.values().first { it.id == dataset }
        }
        val optimizer = extras?.getString("optimizer")
        if (dataset != null) {
            this.optimizer = Optimizers.values().first { it.id == optimizer }
        }
        val learningRate = extras?.getString("learningRate")
        if (learningRate != null) {
            this.learningRate = LearningRates.values().first { it.id == learningRate }
        }
        val momentum = extras?.getString("momentum")
        if (momentum != null) {
            this.momentum = Momentums.values().first { it.id == momentum }
        }
        val l2 = extras?.getString("l2Regularization")
        if (l2 != null) {
            this.l2 = L2Regularizations.values().first { it.id == l2 }
        }
        val batchSize = extras?.getString("batchSize")
        if (batchSize != null) {
            this.batchSize = BatchSizes.values().first { it.id == batchSize }
        }
        val epoch = extras?.getString("epoch")
        if (epoch != null) {
            this.epoch = Epochs.values().first { it.id == epoch }
        }
        val iteratorDistribution = extras?.getString("iteratorDistribution")
        if (iteratorDistribution != null) {
            this.iteratorDistribution =
                IteratorDistributions.values().first { it.id == iteratorDistribution }
        }
        val maxTestSample = extras?.getString("maxTestSample")
        if (maxTestSample != null) {
            this.maxTestSample = MaxTestSamples.values().first { it.id == maxTestSample }
        }
    }

    private fun copyAssets() {
        val assetManager: AssetManager = requireActivity().assets
        for (path in arrayOf("train", "test", "train/Inertial Signals", "test/Inertial Signals")) {
            var files: Array<String>? = null
            try {
                files = assetManager.list(path)
            } catch (e: IOException) {
                logger.error { "Failed to get asset file list." }
            }
            val dir = File(baseDirectory, path)
            if (!dir.exists()) {
                dir.mkdirs()
            }
            if (files != null) {
                for (filename in files) {
                    try {
                        assetManager.open("$path/$filename").use { input ->
                            FileOutputStream(File(dir, filename)).use { output ->
                                copyFile(input, output)
                            }
                        }
                    } catch (e: IOException) {
                        // Probably a directory
                    }
                }
            }
        }
    }

    private fun copyFile(inn: InputStream, out: OutputStream) {
        out.use { fileOut -> inn.copyTo(fileOut) }
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
            seed,
            MLConfiguration(
                dataset = dataset,
                optimizer = optimizer,
                learningRate = learningRate,
                momentum = momentum,
                l2 = l2,
                batchSize = batchSize,
                epoch = epoch,
                iteratorDistribution = iteratorDistribution,
                maxTestSamples = maxTestSample)
        )
    }

    private fun onBtnSimulateDistributedLocallyClicked() {
        SimulatedRunner().run(
            baseDirectory,
            seed,
            MLConfiguration(
                dataset = dataset,
                optimizer = optimizer,
                learningRate = learningRate,
                momentum = momentum,
                l2 = l2,
                batchSize = batchSize,
                epoch = epoch,
                iteratorDistribution = iteratorDistribution,
                maxTestSamples = maxTestSample)
        )
    }

    private fun onBtnRunDistributedClicked() {
        DistributedRunner(getCommunity()).run(
            baseDirectory,
            seed,
            MLConfiguration(
                dataset = dataset,
                optimizer = optimizer,
                learningRate = learningRate,
                momentum = momentum,
                l2 = l2,
                batchSize = batchSize,
                epoch = epoch,
                iteratorDistribution = iteratorDistribution,
                maxTestSamples = maxTestSample)
        )
    }

    override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
        when (parent!!.id) {
            binding.spnDataset.id -> {
                dataset = Datasets.values().first { it.text == datasets[position] }
                setDefaultValuesByDataset(dataset)
                synchronizeSpinners()
            }
            binding.spnOptimizer.id -> optimizer =
                Optimizers.values().first { it.text == optimizers[position] }
            binding.spnLearningRate.id -> learningRate =
                LearningRates.values().first { it.text == learningRates[position] }
            binding.spnMomentum.id -> momentum =
                Momentums.values().first { it.text == momentums[position] }
            binding.spnL2Regularization.id -> l2 =
                L2Regularizations.values().first { it.text == l2Regularizations[position] }
            binding.spnBatchSize.id -> batchSize =
                BatchSizes.values().first { it.text == batchSizes[position] }
            binding.spnEpochs.id -> epoch =
                Epochs.values().first { it.text == epochs[position] }
            binding.spnIteratorDistribution.id -> iteratorDistribution =
                IteratorDistributions.values().first { it.text == iteratorDistributions[position] }
            binding.spnMaxTestSamples.id -> maxTestSample =
                MaxTestSamples.values().first { it.text == maxTestSamples[position] }
        }
    }

    private fun setDefaultValuesByDataset(dataset: Datasets) {
        optimizer = dataset.defaultOptimizer
        learningRate = dataset.defaultLearningRate
        momentum = dataset.defaultMomentum
        l2 = dataset.defaultL2
        batchSize = dataset.defaultBatchSize
        iteratorDistribution = dataset.defaultIteratorDistribution
    }

    private fun synchronizeSpinners(
    ) {
        binding.spnDataset.setSelection(
            datasets.indexOf(dataset.text), true
        )
        binding.spnOptimizer.setSelection(
            optimizers.indexOf(optimizer.text), true
        )
        binding.spnLearningRate.setSelection(
            learningRates.indexOf(learningRate.text), true
        )
        binding.spnMomentum.setSelection(
            momentums.indexOf(momentum.text), true
        )
        binding.spnL2Regularization.setSelection(
            l2Regularizations.indexOf(l2.text), true
        )
        binding.spnBatchSize.setSelection(
            batchSizes.indexOf(batchSize.text), true
        )
        binding.spnEpochs.setSelection(
            epochs.indexOf(epoch.text), true
        )
        binding.spnIteratorDistribution.setSelection(
            iteratorDistributions.indexOf(iteratorDistribution.text), true
        )
        binding.spnMaxTestSamples.setSelection(
            maxTestSamples.indexOf(maxTestSample.text), true
        )
    }

    override fun onNothingSelected(parent: AdapterView<*>?) {
        // Nothing
    }
}
