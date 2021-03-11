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
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.*
import nl.tudelft.trustchain.fedml.databinding.*
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import org.deeplearning4j.common.resources.DL4JResources
import java.io.*

private val logger = KotlinLogging.logger("FedML.MainFragment")
private const val ALWAYS_REPLACE = true

//-e activity fedml -e dataset mnist -e optimizer adam -e learningRate rate_1em3 -e momentum none -e l2Regularization l2_5em3 -e batchSize batch_32 -e maxIteration iter_250 -e iteratorDistribution mnist_1 -e maxTestSample num_200 -e gar mozi -e communicationPattern random -e behavior benign -e runner distributed -e run false
//-e activity fedml -e dataset cifar10 -e optimizer sgd -e learningRate schedule1 -e momentum momentum_1em3 -e l2Regularization l2_1em4 -e batchSize batch_5 -e maxIteration iter_250 -e runner distributed -e run true
//-e activity fedml -e automationPart 0 -e enableExternalAutomation true
class MainFragment : BaseFragment(R.layout.fragment_main), AdapterView.OnItemSelectedListener {
    private val baseDirectory by lazy { requireActivity().filesDir }
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
    private val maxIterations = MaxIterations.values().map { it.text }
    private val iteratorDistributions = IteratorDistributions.values().map { it.text }
    private val maxTestSamples = MaxTestSamples.values().map { it.text }
    private val gars = GARs.values().map { it.text }
    private val communicationPatterns = CommunicationPatterns.values().map { it.text }
    private val behaviors = Behaviors.values().map { it.text }
    private val modelPoisoningAttacks = ModelPoisoningAttacks.values().map { it.text }
    private val numAttackers = NumAttackers.values().map { it.text }

    private var automationPart: Int? = null
    private var dataset = Datasets.MNIST
    private var optimizer = dataset.defaultOptimizer
    private var learningRate = dataset.defaultLearningRate
    private var momentum = dataset.defaultMomentum
    private var l2 = dataset.defaultL2
    private var batchSize = dataset.defaultBatchSize
    private var maxIteration = MaxIterations.ITER_250
    private var iteratorDistribution = dataset.defaultIteratorDistribution
    private var maxTestSample = MaxTestSamples.NUM_20
    private var gar = GARs.BRISTLE
    private var communicationPattern = CommunicationPatterns.RANDOM
    private var behavior = Behaviors.BENIGN
    private var modelPoisoningAttack = ModelPoisoningAttacks.NONE
    private var numAttacker = NumAttackers.NUM_0

    private val community by lazy {
        getIpv8().getOverlay<FedMLCommunity>()
            ?: throw java.lang.IllegalStateException("FedMLCommunity is not configured")
    }
    private val transferRunner by lazy { TransferRunner() }
    private val localRunner by lazy { LocalRunner() }
    private val simulatedRunner by lazy { SimulatedRunner() }
    private val distributedRunner = DistributedRunner(community)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        buttonsBinding.btnTransfer.setOnClickListener { onBtnTransferClicked() }
        buttonsBinding.btnRunLocal.setOnClickListener { onBtnRunLocallyClicked() }
        buttonsBinding.btnRunIsolated.setOnClickListener { onBtnSimulateDistributedLocallyClicked(-1) }
        buttonsBinding.btnRunDistrSim0.setOnClickListener { onBtnSimulateDistributedLocallyClicked(0) }
        buttonsBinding.btnRunDistrSim1.setOnClickListener { onBtnSimulateDistributedLocallyClicked(1) }
        buttonsBinding.btnRunDistrSim2.setOnClickListener { onBtnSimulateDistributedLocallyClicked(2) }
        buttonsBinding.btnRunDistrSim3.setOnClickListener { onBtnSimulateDistributedLocallyClicked(3) }
        buttonsBinding.btnRunDistrSim4.setOnClickListener { onBtnSimulateDistributedLocallyClicked(4) }
        buttonsBinding.btnRunDistrSim5.setOnClickListener { onBtnSimulateDistributedLocallyClicked(5) }
        buttonsBinding.btnRunDistrSim6.setOnClickListener { onBtnSimulateDistributedLocallyClicked(6) }
        buttonsBinding.btnRunDistrSim7.setOnClickListener { onBtnSimulateDistributedLocallyClicked(7) }
        buttonsBinding.btnRunDistrSim8.setOnClickListener { onBtnSimulateDistributedLocallyClicked(8) }
        buttonsBinding.btnRunDistrSim9.setOnClickListener { onBtnSimulateDistributedLocallyClicked(9) }
        buttonsBinding.btnRunDistrSim10.setOnClickListener { onBtnSimulateDistributedLocallyClicked(10) }
        buttonsBinding.btnRunDistrSim11.setOnClickListener { onBtnSimulateDistributedLocallyClicked(11) }
        buttonsBinding.btnRunDistrSim12.setOnClickListener { onBtnSimulateDistributedLocallyClicked(12) }
        buttonsBinding.btnRunDistrSim13.setOnClickListener { onBtnSimulateDistributedLocallyClicked(13) }
        buttonsBinding.btnRunDistrSim14.setOnClickListener { onBtnSimulateDistributedLocallyClicked(14) }
        buttonsBinding.btnRunDistrSim15.setOnClickListener { onBtnSimulateDistributedLocallyClicked(15) }

        bindSpinner(view, datasetBinding.spnDataset, datasets)
        bindSpinner(view, iteratorBinding.spnBatchSize, batchSizes)
        bindSpinner(view, iteratorBinding.spnIteratorDistribution, iteratorDistributions)
        bindSpinner(view, iteratorBinding.spnMaxSamples, maxTestSamples)
        bindSpinner(view, neuralNetworkBinding.spnOptimizer, optimizers)
        bindSpinner(view, neuralNetworkBinding.spnLearningRate, learningRates)
        bindSpinner(view, neuralNetworkBinding.spnMomentum, momentums)
        bindSpinner(view, neuralNetworkBinding.spnL2Regularization, l2Regularizations)
        bindSpinner(view, trainingBinding.spnMaxIteration, maxIterations)
        bindSpinner(view, trainingBinding.spnGar, gars)
        bindSpinner(view, trainingBinding.spnCommunicationPattern, communicationPatterns)
        bindSpinner(view, trainingBinding.spnBehavior, behaviors)
        bindSpinner(view, modelPoisoningBinding.spnAttack, modelPoisoningAttacks)
        bindSpinner(view, modelPoisoningBinding.spnNumAttackers, numAttackers)

        datasetBinding.spnDataset.setSelection(datasets.indexOf(dataset.text))
        iteratorBinding.spnMaxSamples.setSelection(maxTestSamples.indexOf(maxTestSample.text))
        trainingBinding.spnMaxIteration.setSelection(maxIterations.indexOf(maxIteration.text))
        trainingBinding.spnGar.setSelection(gars.indexOf(gar.text))
        trainingBinding.spnCommunicationPattern.setSelection(communicationPatterns.indexOf(communicationPattern.text))
        trainingBinding.spnBehavior.setSelection(behaviors.indexOf(behavior.text))
        modelPoisoningBinding.spnAttack.setSelection(modelPoisoningAttacks.indexOf(modelPoisoningAttack.text))
        modelPoisoningBinding.spnNumAttackers.setSelection(numAttackers.indexOf(numAttacker.text))
        synchronizeSpinners()

        configureDL4JDirectory()
        allowDL4JOnUIThread()
        processIntentExtras()
        copyAssets()

        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0")
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0")

        lifecycleScope.launchWhenStarted {
            while (isActive) {
                updateView()
                delay(500)
            }
        }

        if (requireActivity().intent?.extras?.getString("run") == "true") {
            when (requireActivity().intent?.extras?.getString("runner")) {
                "local" -> onBtnRunLocallyClicked()
//                "simulated" -> onBtnSimulateDistributedLocallyClicked()
//                "distributed" -> onBtnRunDistributedClicked()
                else -> throw IllegalStateException("Runner must be either local, simulated, or distributed")
            }
        }
    }

    private fun updateView() {
        val ipv8 = getIpv8()
        try {
            networkBinding.txtWanAddress.text = community.network.wanLog.estimateWan()!!.toString()
            networkBinding.txtPeers.text = resources.getString(R.string.peers).format(
                ipv8.overlays.values.first { it.javaClass.simpleName == "FedMLCommunity" }.getPeers().size,
                ipv8.overlays.values.first { it.javaClass.simpleName == "UTPCommunity" }.getPeers().size
            )
        } catch (e: NullPointerException) {
            logger.error { "Catched NullPointerException.." }
        }
    }

    private fun processIntentExtras() {
        val extras = requireActivity().intent?.extras

        val automationPart = extras?.getString("automationPart")
        if (automationPart != null) {
            this.automationPart = automationPart.toInt()
        }
        val disableExternalAutomation = extras?.getString("disableExternalAutomation")
        if (disableExternalAutomation == null) {
            distributedRunner.baseDirectory = baseDirectory
            community.enableExternalAutomation(baseDirectory)
        }

        this.dataset = loadDataset(extras?.getString("dataset")) ?: this.dataset
        this.batchSize = loadBatchSize(extras?.getString("batchSize")) ?: this.batchSize
        this.iteratorDistribution = loadIteratorDistribution(extras?.getString("iteratorDistribution")) ?: this.iteratorDistribution
        this.maxTestSample = loadMaxTestSample(extras?.getString("maxTestSample")) ?: this.maxTestSample
        this.optimizer = loadOptimizer(extras?.getString("optimizer")) ?: this.optimizer
        this.learningRate = loadLearningRate(extras?.getString("learningRate")) ?: this.learningRate
        this.momentum = loadMomentum(extras?.getString("momentum")) ?: this.momentum
        this.l2 = loadL2Regularization(extras?.getString("l2Regularization")) ?: this.l2
        this.maxIteration = loadMaxIteration(extras?.getString("maxIteration")) ?: this.maxIteration
        this.gar = loadGAR(extras?.getString("gar")) ?: this.gar
        this.communicationPattern = loadCommunicationPattern(extras?.getString("communicationPattern")) ?: this.communicationPattern
        this.behavior = loadBehavior(extras?.getString("behavior")) ?: this.behavior
        this.modelPoisoningAttack = loadModelPoisoningAttack(extras?.getString("modelPoisoningAttack")) ?: this.modelPoisoningAttack
        this.numAttacker = loadNumAttackers(extras?.getString("numAttackers")) ?: this.numAttacker
    }

    private fun copyAssets() {
        val assetManager = requireActivity().assets
        try {
            val dir = File(baseDirectory, "automation")
            if (!dir.exists()) {
                dir.mkdirs()
            }
            copyAsset(assetManager, "automation.json")
            copyAsset(assetManager, "transfer-mnist")
            copyAsset(assetManager, "transfer-cifar10")
            copyAsset(assetManager, "transfer-wisdm")
        } catch (e: IOException) {
            // Probably a directory
        }
        for (path in arrayOf("train", "test", "train/Inertial Signals", "test/Inertial Signals")) {
            val files = assetManager.list(path)!!
            val dir = File(baseDirectory, path)
            if (!dir.exists()) {
                dir.mkdirs()
            }
            for (filename in files) {
                try {
                    copyAsset(assetManager, "$path/$filename")
                    /*assetManager.open("$path/$filename").use { input ->
                        FileOutputStream(File(dir, filename)).use { output ->
                            copyFile(input, output)
                        }
                    }*/
                } catch (e: IOException) {
                    // Probably a directory
                }
            }
        }
    }

    private fun copyAsset(assetManager: AssetManager, asset: String) {
        val file = File(baseDirectory, asset)
        if (!file.exists() || ALWAYS_REPLACE) {
            assetManager.open(asset).use { input ->
                FileOutputStream(file).use { output ->
                    copyFile(input, output)
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
        val policy = StrictMode.ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)
    }

    ////// BUTTON CLICK LISTENERS

    private fun onBtnTransferClicked() {
        transferRunner.run(
            baseDirectory,
            getSeed(),
            createMLConfiguration()
        )
    }

    private fun onBtnRunLocallyClicked() {
        localRunner.run(
            baseDirectory,
            getSeed(),
            createMLConfiguration()
        )
    }

    private fun onBtnSimulateDistributedLocallyClicked(automationPart: Int) {
//        if (automationPart != null) {
            simulatedRunner.simulate(
                baseDirectory,
                automationPart
            )
//        } else {
//            simulatedRunner.run(
//                baseDirectory,
//                getSeed(),
//                createMLConfiguration()
//            )
//        }
    }

    private fun getSeed(): Int {
        return community.myEstimatedWan.port
    }

    private fun createMLConfiguration(): MLConfiguration {
        return MLConfiguration(
            dataset,
            DatasetIteratorConfiguration(
                batchSize = batchSize,
                maxTestSamples = maxTestSample,
                distribution = iteratorDistribution.value.toList()
            ),
            NNConfiguration(
                optimizer = optimizer,
                learningRate = learningRate,
                momentum = momentum,
                l2 = l2
            ),
            TrainConfiguration(
                maxIteration = maxIteration,
                gar = gar,
                communicationPattern = communicationPattern,
                behavior = behavior,
                slowdown = Slowdowns.NONE,
                joiningLate = TransmissionRounds.N0
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
                MaxTestSamples.values().first { it.text == maxTestSamples[position] }
            neuralNetworkBinding.spnOptimizer.id -> optimizer =
                Optimizers.values().first { it.text == optimizers[position] }
            neuralNetworkBinding.spnLearningRate.id -> learningRate =
                LearningRates.values().first { it.text == learningRates[position] }
            neuralNetworkBinding.spnMomentum.id -> momentum =
                Momentums.values().first { it.text == momentums[position] }
            neuralNetworkBinding.spnL2Regularization.id -> l2 =
                L2Regularizations.values().first { it.text == l2Regularizations[position] }
            trainingBinding.spnMaxIteration.id -> maxIteration =
                MaxIterations.values().first { it.text == maxIterations[position] }
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
        trainingBinding.spnMaxIteration.setSelection(
            maxIterations.indexOf(maxIteration.text), true
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
