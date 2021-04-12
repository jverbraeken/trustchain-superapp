package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.ipv8.Peer
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaClientToServer
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaServerToClient
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread
import kotlin.random.Random

private val logger = KotlinLogging.logger("SimulatedRunner")

class SimulatedRunner : Runner() {
    private lateinit var nodes: List<Node>
    private var peersRR: MutableMap<Int, MutableList<Node>?> = HashMap()
    private var peersRing: MutableMap<Int, MutableList<Node>?> = HashMap()
    private var ringCounter: MutableMap<Int, Int> = HashMap()

    override fun run(baseDirectory: File, _unused: Int, _unused2: MLConfiguration) {
        simulate(baseDirectory, 0)
    }

    fun simulate(
        baseDirectory: File,
        automationPart: Int,
    ) {
        val job = SupervisorJob()
        val scope = CoroutineScope(Dispatchers.Default + job)
        scope.launch {
        val evaluationProcessor = EvaluationProcessor(
            baseDirectory,
            "simulated",
            listOf(
                "before or after averaging",
                "#peers included in current batch"
            )
        )
        try {
            val automation = loadAutomation(baseDirectory)
            logger.debug { "Automation loaded" }
            val (configs, figureNames) = generateConfigs(automation, automationPart)
            logger.debug { "Configs generated" }

            for (figure in configs.indices) {
                val figureName = figureNames[figure]
                val figureConfig = configs[figure]
                for (test in figureConfig.indices) {
                    val testConfig = figureConfig[test]
                    performTest(baseDirectory, figureName, testConfig, evaluationProcessor)
                }
            }
            evaluationProcessor.done()
            logger.error { "All tests finished" }
        } catch (e: Exception) {
            evaluationProcessor.error(e)
            e.printStackTrace()
        }
        }
    }

    private fun performTest(
        baseDirectory: File,
        figureName: String,
        testConfig: List<MLConfiguration>,
        evaluationProcessor: EvaluationProcessor
    ) {
        val transfer = testConfig[0].trainConfiguration.transfer
        val fullFigureName = "$figureName - ${testConfig[0].trainConfiguration.gar.id} - ${if (transfer) "transfer" else "regular"}"
        logger.error { "Going to test: $fullFigureName" }

        // Initialize everything
        evaluationProcessor.newSimulation(fullFigureName, testConfig, transfer)
        val start = System.currentTimeMillis()
        nodes = testConfig.mapIndexed { i, config ->
            Node(
                i,
                config,
                ::generateNetwork,
                ::getDataSetIterators,
                baseDirectory,
                evaluationProcessor,
                start,
                ::shareModel
            )
        }
        testConfig.forEachIndexed { i, _ ->
            ringCounter[i] = 1
        }
        val countPerPeers = getCountPerPeers(testConfig, nodes)
        nodes.forEachIndexed { i, node -> node.setCountPerPeer(countPerPeers.getValue(i)) }
        nodes[0].printIterations()

        // Perform <x> iterations
        var epochEnd = true
        var epoch = -1
        for (iteration in 0 until testConfig[0].trainConfiguration.maxIteration.value) {
            if (epochEnd) {
                epoch++
                logger.debug { "Epoch: $epoch" }
                epochEnd = false
            }
            logger.debug { "Iteration: $iteration" }

            nodes.forEach { it.applyNetworkBuffers() }
            val endEpochs = nodes.mapIndexed { i, node ->
                node.performIteration(epoch, iteration)
            }
            if (endEpochs.any { it }) epochEnd = true
        }
        logger.warn { "Test finished" }
    }

    private fun getCountPerPeers(testConfig: List<MLConfiguration>, nodes: List<Node>): Map<Int, Map<Int, Int>> {
        val toServerMessageBuffers = testConfig.map { CopyOnWriteArrayList<MsgPsiCaClientToServer>() }.toTypedArray()
        val toClientMessageBuffers = testConfig.map { CopyOnWriteArrayList<MsgPsiCaServerToClient>() }.toTypedArray()
        val countPerPeers = ConcurrentHashMap<Int, Map<Int, Int>>()
        val threads = testConfig.mapIndexed { i, _ ->
            thread {
                countPerPeers[i] = getSimilarPeers(
                    nodes[i].getLabels(),
                    nodes[i].getSRAKeyPair(),
                    toServerMessageBuffers,
                    toClientMessageBuffers,
                    i
                )
            }
        }
        threads.forEach { it.join() }
        return countPerPeers
    }

    private fun getSimilarPeers(
        labels: List<String>,
        sraKeyPair: SRAKeyPair,
        toServerMessageBuffers: Array<CopyOnWriteArrayList<MsgPsiCaClientToServer>>,
        toClientMessageBuffers: Array<CopyOnWriteArrayList<MsgPsiCaServerToClient>>,
        i: Int,
    ): Map<Int, Int> {
        val encryptedLabels = clientsRequestsServerLabels(
            labels,
            sraKeyPair
        )
        toServerMessageBuffers
            .filterIndexed { index, _ -> index != i }
            .forEach { it.add(MsgPsiCaClientToServer(encryptedLabels, i)) }

        while (toServerMessageBuffers[i].size < toServerMessageBuffers.size - 1) {
            Thread.sleep(1)
        }

        for (toServerMessage in toServerMessageBuffers[i]) {
            val (reEncryptedLabels, filter) = serverRespondsClientRequests(
                labels,
                toServerMessage,
                sraKeyPair
            )
            val message = MsgPsiCaServerToClient(reEncryptedLabels, filter, i)
            toClientMessageBuffers[toServerMessage.client].add(message)
        }

        while (toClientMessageBuffers[i].size < toClientMessageBuffers.size - 1) {
            Thread.sleep(1)
        }

        return clientReceivesServerResponses(
            i,
            toClientMessageBuffers[i],
            sraKeyPair
        )
    }

    private fun shareModel(
        params: INDArray,
        trainConfiguration: TrainConfiguration,
        random: Random,
        nodeIndex: Int,
        countPerPeer: Map<Int, Int>
    ) {
        val message = craftMessage(params, trainConfiguration.behavior, random)
        when (trainConfiguration.communicationPattern) {
            CommunicationPatterns.ALL -> nodes
                .filter { it.getNodeIndex() != nodeIndex && (if (trainConfiguration.gar == GARs.BRISTLE) it.getNodeIndex() in countPerPeer.keys else true) }
                .forEach { it.addNetworkMessage(nodeIndex, message) }
            CommunicationPatterns.RANDOM -> nodes
                .filter { it.getNodeIndex() != nodeIndex && (if (trainConfiguration.gar == GARs.BRISTLE) it.getNodeIndex() in countPerPeer.keys else true) }
                .random().addNetworkMessage(nodeIndex, message)
            CommunicationPatterns.RR -> {
                if (peersRR[nodeIndex].isNullOrEmpty()) {
                    peersRR[nodeIndex] = nodes.filter { it.getNodeIndex() != nodeIndex && (if (trainConfiguration.gar == GARs.BRISTLE) (it.getNodeIndex() in countPerPeer && countPerPeer[it.getNodeIndex()]!! >= 1) else true) }.toMutableList()
                    val index = peersRR[nodeIndex]!!.indexOfFirst { it.getNodeIndex() > nodeIndex }
                    for (i in 0 until index) {
                        peersRR[nodeIndex]!!.add(peersRR[nodeIndex]!!.removeAt(0))
                    }
                }
                peersRR[nodeIndex]!!.removeAt(0).addNetworkMessage(nodeIndex, message)
            }
            CommunicationPatterns.RING -> {
                if (peersRing[nodeIndex].isNullOrEmpty() || peersRing[nodeIndex]!!.size < ringCounter[nodeIndex]!!) {
                    peersRing[nodeIndex] = nodes.filter { it.getNodeIndex() != nodeIndex && (if (trainConfiguration.gar == GARs.BRISTLE) (it.getNodeIndex() in countPerPeer && countPerPeer[it.getNodeIndex()]!! >= 1) else true) }.toMutableList()
                    val index = peersRing[nodeIndex]!!.indexOfFirst { it.getNodeIndex() > nodeIndex }
                    for (i in 0 until index) {
                        peersRing[nodeIndex]!!.add(peersRing[nodeIndex]!!.removeAt(0))
                    }
                    ringCounter[nodeIndex] = 1
                }
                for (i in 0 until ringCounter[nodeIndex]!! - 1) {
                    peersRing[nodeIndex]!!.removeAt(0)
                }
                ringCounter[nodeIndex] = ringCounter[nodeIndex]!! * 2
                peersRing[nodeIndex]!!.removeAt(0).addNetworkMessage(nodeIndex, message)
            }
            CommunicationPatterns.RANDOM_3 -> {
                repeat(3) {
                    nodes
                        .filter { it.getNodeIndex() != nodeIndex && (if (trainConfiguration.gar == GARs.BRISTLE) it.getNodeIndex() in countPerPeer.keys else true) }
                        .random().addNetworkMessage(nodeIndex, message)
                }
            }
        }
    }
}
