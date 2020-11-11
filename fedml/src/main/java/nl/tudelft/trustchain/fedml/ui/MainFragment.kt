package nl.tudelft.trustchain.fedml.ui

import android.content.res.AssetManager
import android.os.Bundle
import android.os.StrictMode
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import mu.KotlinLogging
import nl.tudelft.trustchain.common.ui.BaseFragment
import nl.tudelft.trustchain.common.util.viewBinding
import nl.tudelft.trustchain.fedml.R
import nl.tudelft.trustchain.fedml.ai.*
import nl.tudelft.trustchain.fedml.databinding.*
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import nl.tudelft.trustchain.fedml.ipv8.MsgPing
import org.deeplearning4j.common.resources.DL4JResources
import java.io.*

private val logger = KotlinLogging.logger("FedML.MainFragment")

//-e activity fedml -e dataset mnist -e optimizer adam -e learningRate rate_1em3 -e momentum none -e l2Regularization l2_5em3 -e batchSize batch_32 -e epoch epoch_50 -e iteratorDistribution mnist_1 -e maxTestSample num_200 -e gar mozi -e communicationPattern random -e behavior benign -e runner distributed -e run false
//-e activity fedml -e dataset cifar10 -e optimizer sgd -e learningRate schedule1 -e momentum momentum_1em3 -e l2Regularization l2_1em4 -e batchSize batch_5 -e epoch epoch_25 -e runner distributed -e run true
class MainFragment : BaseFragment(R.layout.fragment_main), AdapterView.OnItemSelectedListener {
    private val baseDirectory: File by lazy { requireActivity().filesDir }
    private val networkBinding by viewBinding(FragmentMainNetworkBinding::bind)
    private val buttonsBinding by viewBinding(FragmentMainButtonsBinding::bind)
    private val datasetBinding by viewBinding(FragmentMainDatasetBinding::bind)
    private val iteratorBinding by viewBinding(FragmentMainIteratorBinding::bind)
    private val neuralNetworkBinding by viewBinding(FragmentMainNeuralNetworkBinding::bind)
    private val trainingBinding by viewBinding(FragmentMainTrainingBinding::bind)
    private val modelPoisoningBinding by viewBinding(FragmentMainModelPoisoningBinding::bind)

    private val datasets = Datasets.values().map { it.text }
    private val optimizers = Optimizers.values().map { it.text }
    private val learningRates = LearningRates.values().map { it.text }
    private val momentums = Momentums.values().map { it.text }
    private val l2Regularizations = L2Regularizations.values().map { it.text }
    private val batchSizes = BatchSizes.values().map { it.text }
    private val epochs = Epochs.values().map { it.text }
    private val iteratorDistributions = IteratorDistributions.values().map { it.text }
    private val maxTestSamples = MaxSamples.values().map { it.text }
    private val gars = GARs.values().map { it.text }
    private val communicationPatterns = CommunicationPatterns.values().map { it.text }
    private val behaviors = Behaviors.values().map { it.text }
    private val modelPoisoningAttacks = ModelPoisoningAttacks.values().map { it.text }
    private val numAttackers = NumAttackers.values().map { it.text }

    private var dataset: Datasets = Datasets.MNIST
    private var optimizer: Optimizers = dataset.defaultOptimizer
    private var learningRate: LearningRates = dataset.defaultLearningRate
    private var momentum: Momentums = dataset.defaultMomentum
    private var l2: L2Regularizations = dataset.defaultL2
    private var batchSize: BatchSizes = dataset.defaultBatchSize
    private var epoch: Epochs = Epochs.EPOCH_5
    private var iteratorDistribution: IteratorDistributions = dataset.defaultIteratorDistribution
    private var maxTestSample = MaxSamples.NUM_40
    private var gar = GARs.BRISTLE
    private var communicationPattern = CommunicationPatterns.RANDOM
    private var behavior = Behaviors.BENIGN
    private var modelPoisoningAttack = ModelPoisoningAttacks.NONE
    private var numAttacker = NumAttackers.NUM_2

    private fun getCommunity(): FedMLCommunity {
        return getIpv8().getOverlay()
            ?: throw java.lang.IllegalStateException("FedMLCommunity is not configured")
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        buttonsBinding.btnPing.setOnClickListener { onBtnPingClicked() }
        buttonsBinding.btnRunLocal.setOnClickListener { onBtnRunLocallyClicked() }
        buttonsBinding.btnRunDistrSim.setOnClickListener { onBtnSimulateDistributedLocallyClicked() }
        buttonsBinding.btnRunDistr.setOnClickListener { onBtnRunDistributedClicked() }

        bindSpinner(view, datasetBinding.spnDataset, datasets)
        bindSpinner(view, iteratorBinding.spnBatchSize, batchSizes)
        bindSpinner(view, iteratorBinding.spnIteratorDistribution, iteratorDistributions)
        bindSpinner(view, iteratorBinding.spnMaxSamples, maxTestSamples)
        bindSpinner(view, neuralNetworkBinding.spnOptimizer, optimizers)
        bindSpinner(view, neuralNetworkBinding.spnLearningRate, learningRates)
        bindSpinner(view, neuralNetworkBinding.spnMomentum, momentums)
        bindSpinner(view, neuralNetworkBinding.spnL2Regularization, l2Regularizations)
        bindSpinner(view, trainingBinding.spnEpochs, epochs)
        bindSpinner(view, trainingBinding.spnGar, gars)
        bindSpinner(view, trainingBinding.spnCommunicationPattern, communicationPatterns)
        bindSpinner(view, trainingBinding.spnBehavior, behaviors)
        bindSpinner(view, modelPoisoningBinding.spnAttack, modelPoisoningAttacks)
        bindSpinner(view, modelPoisoningBinding.spnNumAttackers, numAttackers)

        configureDL4JDirectory()
        allowDL4JOnUIThread()
        datasetBinding.spnDataset.setSelection(datasets.indexOf(dataset.text))
        iteratorBinding.spnMaxSamples.setSelection(maxTestSamples.indexOf(maxTestSample.text))
        trainingBinding.spnEpochs.setSelection(epochs.indexOf(epoch.text))
        trainingBinding.spnGar.setSelection(gars.indexOf(gar.text))
        trainingBinding.spnCommunicationPattern.setSelection(communicationPatterns.indexOf(communicationPattern.text))
        trainingBinding.spnBehavior.setSelection(behaviors.indexOf(behavior.text))
        modelPoisoningBinding.spnAttack.setSelection(modelPoisoningAttacks.indexOf(modelPoisoningAttack.text))
        modelPoisoningBinding.spnNumAttackers.setSelection(numAttackers.indexOf(numAttacker.text))
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

        lifecycleScope.launchWhenStarted {
            while (isActive) {
                updateView()
                delay(500)
            }
        }
    }

    private fun updateView() {
        val ipv8 = getIpv8()
        val demo = getDemoCommunity()
        networkBinding.txtWanAddress.text = demo.myEstimatedWan.toString()
        networkBinding.txtPeers.text = resources.getString(R.string.peers).format(
            ipv8.overlays.values.first { it.javaClass.simpleName == "FedMLCommunity" }.getPeers().size,
            ipv8.overlays.values.first { it.javaClass.simpleName == "UTPCommunity" }.getPeers().size
        )
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
            this.maxTestSample = MaxSamples.values().first { it.id == maxTestSample }
        }
        val gar = extras?.getString("gar")
        if (gar != null) {
            this.gar = GARs.values().first { it.id == gar }
        }
        val communicationPattern = extras?.getString("communicationPattern")
        if (communicationPattern != null) {
            this.communicationPattern = CommunicationPatterns.values().first { it.id == communicationPattern }
        }
        val behavior = extras?.getString("behavior")
        if (behavior != null) {
            this.behavior = Behaviors.values().first { it.id == behavior }
        }
        val modelPoisoningAttack = extras?.getString("modelPoisoningAttack")
        if (modelPoisoningAttack != null) {
            this.modelPoisoningAttack = ModelPoisoningAttacks.values().first { it.id == modelPoisoningAttack }
        }
        val numAttackers = extras?.getString("numAttackers")
        if (numAttackers != null) {
            this.numAttacker = NumAttackers.values().first { it.id == numAttackers }
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
            getSeed(),
            createMLConfiguration()
        )
    }

    private fun onBtnSimulateDistributedLocallyClicked() {
        SimulatedRunner().run(
            baseDirectory,
            getSeed(),
            createMLConfiguration()
        )
    }

    private fun onBtnRunDistributedClicked() {
        DistributedRunner(getCommunity()).run(
            baseDirectory,
            getSeed(),
            createMLConfiguration()
        )
    }

    private fun getSeed(): Int {
        return getCommunity().myEstimatedWan.port
    }

    private fun createMLConfiguration(): MLConfiguration {
        return MLConfiguration(
            dataset,
            DatasetIteratorConfiguration(
                batchSize = batchSize,
                maxTestSamples = maxTestSample,
                distribution = iteratorDistribution
            ),
            NNConfiguration(
                optimizer = optimizer,
                learningRate = learningRate,
                momentum = momentum,
                l2 = l2
            ),
            TrainConfiguration(
                numEpochs = epoch,
                gar = gar,
                communicationPattern = communicationPattern,
                behavior = behavior
            ),
            ModelPoisoningConfiguration(
                attack = modelPoisoningAttack,
                numAttackers = numAttacker
            )
        )
    }

    override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
        when (parent!!.id) {
            datasetBinding.spnDataset.id -> {
                dataset = Datasets.values().first { it.text == datasets[position] }
                setDefaultValuesByDataset(dataset)
                synchronizeSpinners()
            }
            iteratorBinding.spnBatchSize.id -> batchSize =
                BatchSizes.values().first { it.text == batchSizes[position] }
            iteratorBinding.spnIteratorDistribution.id -> iteratorDistribution =
                IteratorDistributions.values().first { it.text == iteratorDistributions[position] }
            iteratorBinding.spnMaxSamples.id -> maxTestSample =
                MaxSamples.values().first { it.text == maxTestSamples[position] }
            neuralNetworkBinding.spnOptimizer.id -> optimizer =
                Optimizers.values().first { it.text == optimizers[position] }
            neuralNetworkBinding.spnLearningRate.id -> learningRate =
                LearningRates.values().first { it.text == learningRates[position] }
            neuralNetworkBinding.spnMomentum.id -> momentum =
                Momentums.values().first { it.text == momentums[position] }
            neuralNetworkBinding.spnL2Regularization.id -> l2 =
                L2Regularizations.values().first { it.text == l2Regularizations[position] }
            trainingBinding.spnEpochs.id -> epoch =
                Epochs.values().first { it.text == epochs[position] }
            trainingBinding.spnGar.id -> {
                gar = GARs.values().first { it.text == gars[position] }
                modelPoisoningAttack = gar.defaultModelPoisoningAttack
                synchronizeSpinners()
            }
            trainingBinding.spnCommunicationPattern.id -> communicationPattern =
                CommunicationPatterns.values().first { it.text == communicationPatterns[position] }
            trainingBinding.spnBehavior.id -> behavior =
                Behaviors.values().first { it.text == behaviors[position] }
            modelPoisoningBinding.spnAttack.id -> modelPoisoningAttack =
                ModelPoisoningAttacks.values().first { it.text == modelPoisoningAttacks[position] }
            modelPoisoningBinding.spnNumAttackers.id -> numAttacker =
                NumAttackers.values().first { it.text == numAttackers[position] }
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
        datasetBinding.spnDataset.setSelection(
            datasets.indexOf(dataset.text), true
        )
        iteratorBinding.spnBatchSize.setSelection(
            batchSizes.indexOf(batchSize.text), true
        )
        iteratorBinding.spnIteratorDistribution.setSelection(
            iteratorDistributions.indexOf(iteratorDistribution.text), true
        )
        iteratorBinding.spnMaxSamples.setSelection(
            maxTestSamples.indexOf(maxTestSample.text), true
        )
        neuralNetworkBinding.spnOptimizer.setSelection(
            optimizers.indexOf(optimizer.text), true
        )
        neuralNetworkBinding.spnLearningRate.setSelection(
            learningRates.indexOf(learningRate.text), true
        )
        neuralNetworkBinding.spnMomentum.setSelection(
            momentums.indexOf(momentum.text), true
        )
        neuralNetworkBinding.spnL2Regularization.setSelection(
            l2Regularizations.indexOf(l2.text), true
        )
        trainingBinding.spnEpochs.setSelection(
            epochs.indexOf(epoch.text), true
        )
        trainingBinding.spnGar.setSelection(
            gars.indexOf(gar.text), true
        )
        trainingBinding.spnCommunicationPattern.setSelection(
            communicationPatterns.indexOf(communicationPattern.text), true
        )
        trainingBinding.spnBehavior.setSelection(
            behaviors.indexOf(behavior.text), true
        )
        modelPoisoningBinding.spnAttack.setSelection(
            modelPoisoningAttacks.indexOf(modelPoisoningAttack.text), true
        )
        modelPoisoningBinding.spnNumAttackers.setSelection(
            numAttackers.indexOf(numAttacker.text), true
        )
    }

    override fun onNothingSelected(parent: AdapterView<*>?) {
        // Nothing
    }
}
