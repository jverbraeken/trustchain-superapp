package nl.tudelft.trustchain.fedml.ipv8

import com.google.common.hash.BloomFilter
import com.google.common.primitives.Longs
import mu.KotlinLogging
import nl.tudelft.ipv8.IPv4Address
import nl.tudelft.ipv8.Overlay
import nl.tudelft.ipv8.Peer
import nl.tudelft.ipv8.attestation.trustchain.TrustChainCommunity
import nl.tudelft.ipv8.attestation.trustchain.TrustChainCrawler
import nl.tudelft.ipv8.attestation.trustchain.TrustChainSettings
import nl.tudelft.ipv8.attestation.trustchain.store.TrustChainStore
import nl.tudelft.ipv8.keyvault.defaultCryptoProvider
import nl.tudelft.ipv8.messaging.Deserializable
import nl.tudelft.ipv8.messaging.Packet
import nl.tudelft.ipv8.messaging.Serializable
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.*
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.*
import java.math.BigInteger
import kotlin.concurrent.thread

private val logger = KotlinLogging.logger("FedMLCommunity")

interface MessageListener {
    fun onMessageReceived(messageId: FedMLCommunity.MessageId, peer: Peer, payload: Any)
}

class FedMLCommunity(
    settings: TrustChainSettings,
    database: TrustChainStore,
    crawler: TrustChainCrawler = TrustChainCrawler(),
) : TrustChainCommunity(settings, database, crawler) {
    override val serviceId = "36b098237ff4debfd0278b8b87c583e1c2cce4b7"
    private val masterAddress = IPv4Address("10.0.2.2", 55555)
    private var master: Peer = Peer(defaultCryptoProvider.generateKey(), masterAddress, supportsUTP = true)
    private var peersRR: MutableList<Peer>? = null
    private var peersRing: MutableList<Peer>? = null
    private var ringCounter = 1

    class Factory(
        private val settings: TrustChainSettings,
        private val database: TrustChainStore,
        private val crawler: TrustChainCrawler = TrustChainCrawler(),
    ) : Overlay.Factory<FedMLCommunity>(FedMLCommunity::class.java) {
        override fun create(): FedMLCommunity {
            return FedMLCommunity(settings, database, crawler)
        }
    }

    // I'm claiming range 100 - 120
    // TODO: we should really create a class for the whole superapp project that manages the generation of unique message IDs
    enum class MessageId(val id: Int, val deserializer: Deserializable<out Any>) {
        MSG_PING(100, MsgPing.Deserializer),
        MSG_PONG(101, MsgPong.Deserializer),
        MSG_PARAM_UPDATE(102, MsgParamUpdate.Deserializer),
        MSG_PSI_CA_CLIENT_TO_SERVER(103, MsgPsiCaClientToServer.Deserializer),
        MSG_PSI_CA_SERVER_TO_CLIENT(104, MsgPsiCaServerToClient.Deserializer),
        MSG_NOTIFY_HEARTBEAT(110, MsgNotifyHeartbeat.Deserializer),
        MSG_NEW_TEST_COMMAND(111, MsgNewTestCommand.Deserializer),
        MSG_NOTIFY_EVALUATION(112, MsgNotifyEvaluation.Deserializer),
        MSG_NOTIFY_FINISHED(113, MsgNotifyFinished.Deserializer)
    }

    init {
        messageHandlers[MessageId.MSG_PING.id] = ::onMsgPing
        messageHandlers[MessageId.MSG_PONG.id] = ::onMsgPong
        messageHandlers[MessageId.MSG_PARAM_UPDATE.id] = ::onMsgParamUpdate
        messageHandlers[MessageId.MSG_PSI_CA_CLIENT_TO_SERVER.id] = ::onMsgPsiCaClientToServer
        messageHandlers[MessageId.MSG_PSI_CA_SERVER_TO_CLIENT.id] = ::onMsgPsiCaServerToClient
        messageHandlers[MessageId.MSG_NEW_TEST_COMMAND.id] = ::onMsgNewTestCommand

        messageListeners[MessageId.MSG_PING]!!.add(object : MessageListener {
            override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
                sendToPeer(peer, MessageId.MSG_PONG, MsgPong("Pong"), reliable = true)
            }
        })
    }

    companion object {
        val messageListeners = MessageId.values().associate { it to mutableListOf<MessageListener>() }.toMutableMap()
        fun registerMessageListener(messageId: MessageId, listener: MessageListener) {
            messageListeners[messageId]!!.add(listener)
        }
    }

    fun enableExternalAutomation(baseDirectory: File) {
        writePortNumber(baseDirectory)
        keepSendingHeartbeats()
    }

    private fun writePortNumber(baseDirectory: File) = thread {
        while (myEstimatedWan.port == 0) {
            Thread.sleep(100)
        }
        val wanPort = myEstimatedWan.port
        val file = File(baseDirectory, "wanPort")
        file.delete()
        file.createNewFile()
        PrintWriter(file).use {
            it.println(wanPort)
        }
    }

    private fun keepSendingHeartbeats() = thread {
        while (true) {
            sendToMaster(MessageId.MSG_NOTIFY_HEARTBEAT, MsgNotifyHeartbeat(true))
            Thread.sleep(2000)
        }
    }

    internal fun sendToMaster(messageID: MessageId, message: Serializable, logging: Boolean = false, reliable: Boolean = false) {
        logger.debug { "sendToMaster, messageId: ${messageID.id}" }
        val packet = serializePacket(messageID.id, message, true, logging = logging)
        send(master, packet, reliable)
    }

    internal fun sendToPeer(
        peer: Peer,
        messageID: MessageId,
        message: Serializable,
        logging: Boolean = false,
        reliable: Boolean = false,
    ) {
        logger.debug { "sendToPeer, messageId: ${messageID.id}" }
        val packet = serializePacket(messageID.id, message, true, logging = logging)
        peer.supportsUTP = true
        send(peer, packet, reliable)
    }

    internal fun sendToAll(
        messageID: MessageId,
        message: Serializable,
        priorityPeers: Set<Int>? = null,
        reliable: Boolean = false,
    ) {
        logger.debug { "sendToAll" }
        for (peer in getAllowedPeers(priorityPeers)) {
            logger.debug { "Peer: ${peer.address}" }
            sendToPeer(peer, messageID, message, reliable = reliable)
        }
    }

    internal fun sendToRandomPeer(
        messageID: MessageId,
        message: Serializable,
        priorityPeers: Set<Int>? = null,
        reliable: Boolean = false,
    ) {
        logger.debug { "sendToRandomPeer" }
        val peers = getAllowedPeers(priorityPeers)
        if (peers.isNotEmpty()) {
            val peer = peers.random()
            logger.debug { "Peer: ${peer.address}" }
            sendToPeer(peer, messageID, message, reliable = reliable)
        }
    }

    private fun getAllowedPeers(peers: Set<Int>?): List<Peer> {
        return getPeers().filter { if (peers == null) true else it.address.port in peers }
    }

    // Round Robin
    internal fun sendToNextPeerRR(
        messageID: MessageId,
        message: Serializable,
        priorityPeers: Set<Int>? = null,
        reliable: Boolean = false,
    ) {
        logger.debug { "sendToNextPeerRR" }
        if (!nl.tudelft.ipv8.messaging.utp.canSend()) {
            logger.debug { "Skipped because busy sending" }
            return
        }
        val peer = getAndSetNextPeerRR(priorityPeers)
        if (peer != null) {
            logger.debug { "Peer: ${peer.address}" }
            sendToPeer(peer, messageID, message, reliable = reliable)
        }
    }

    private fun getAndSetNextPeerRR(priorityPeers: Set<Int>? = null): Peer? {
        if (peersRR.isNullOrEmpty()) {
            peersRR = getAllowedPeers(priorityPeers).toMutableList()
        }
        return if (peersRR!!.isEmpty()) {
            logger.debug { "No peer found" }
            null
        } else {
            peersRR!!.removeAt(0)
        }
    }

    internal fun sendToNextPeerRing(
        messageID: MessageId,
        message: Serializable,
        priorityPeers: Set<Int>? = null,
        reliable: Boolean = false,
    ) {
        logger.debug { "sendToNextPeerRing" }
        if (!nl.tudelft.ipv8.messaging.utp.canSend()) {
            logger.debug { "Skipped because busy sending" }
            return
        }
        val peer = getAndSetNextPeerRing(priorityPeers)
        if (peer != null) {
            logger.debug { "Peer: ${peer.address}" }
            sendToPeer(peer, messageID, message, reliable = reliable)
        }
    }

    private fun getAndSetNextPeerRing(priorityPeers: Set<Int>? = null): Peer? {
        if (peersRing.isNullOrEmpty() || peersRing!!.size < ringCounter) {
            peersRing = getAllowedPeers(priorityPeers).toMutableList()
            peersRing!!.sortBy { it.address.port }
            val index = peersRing!!.indexOfFirst { it.address.port > myEstimatedWan.port }
            for (i in 0 until index) {
                val peer = peersRing!!.removeAt(0)
                peersRing!!.add(peer)
            }
            logger.debug { "peersRing = { ${peersRing!!.map { it.address.port }.joinToString(", ")}" }
            ringCounter = 1
        }
        return if (peersRing!!.isEmpty()) {
            logger.debug { "No peers found" }
            null
        } else {
            for (i in 0 until ringCounter - 1) {
                peersRing!!.removeAt(0)
                logger.debug { "peersRing after removal = { ${peersRing!!.map { it.address.port }.joinToString(", ")}" }
            }
            ringCounter *= 2
            peersRing!!.removeAt(0)
        }
    }

    ////// MESSAGE RECEIVED EVENTS

    private fun onMsgPing(packet: Packet) {
        onMessage(packet, MessageId.MSG_PING)
    }

    private fun onMsgPong(packet: Packet) {
        onMessage(packet, MessageId.MSG_PONG)
    }

    private fun onMsgParamUpdate(packet: Packet) {
        onMessage(packet, MessageId.MSG_PARAM_UPDATE)
    }

    private fun onMsgPsiCaClientToServer(packet: Packet) {
        onMessage(packet, MessageId.MSG_PSI_CA_CLIENT_TO_SERVER)
    }

    private fun onMsgPsiCaServerToClient(packet: Packet) {
        onMessage(packet, MessageId.MSG_PSI_CA_SERVER_TO_CLIENT)
    }

    private fun onMsgNewTestCommand(packet: Packet) {
        onMessage(packet, MessageId.MSG_NEW_TEST_COMMAND)
    }

    private fun onMessage(packet: Packet, messageId: MessageId) {
        val (peer, payload) = packet.getAuthPayload(messageId.deserializer)
        logger.debug { "${messageId.name}: ${peer.mid}" }
        messageListeners[messageId]!!.forEach { it.onMessageReceived(messageId, peer, payload) }
    }
}

////// MESSAGE DATA CLASSES

data class MsgPing(val message: String) : Serializable {
    override fun serialize(): ByteArray {
        return message.toByteArray()
    }

    companion object Deserializer : Deserializable<MsgPing> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgPing, Int> {
            val message = buffer.toString(Charsets.UTF_8)
            return Pair(MsgPing(message), buffer.size)
        }
    }
}

data class MsgPong(val message: String) : Serializable {
    override fun serialize(): ByteArray {
        return message.toByteArray()
    }

    companion object Deserializer : Deserializable<MsgPong> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgPong, Int> {
            val message = buffer.toString(Charsets.UTF_8)
            return Pair(MsgPong(message), buffer.size)
        }
    }
}

data class MsgParamUpdate(val array: INDArray) : Serializable {
    override fun serialize(): ByteArray {
        return ByteArrayOutputStream().use { ObjectOutputStream(it).writeObject(array); it }.toByteArray()
    }

    companion object Deserializer : Deserializable<MsgParamUpdate> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgParamUpdate, Int> {
            val croppedBuffer = buffer.copyOfRange(offset, buffer.size)
            ByteArrayInputStream(croppedBuffer).use { bis ->
                ObjectInputStream(bis).use { ois ->
                    return Pair(MsgParamUpdate(ois.readObject() as INDArray), buffer.size)
                }
            }
        }
    }
}

data class MsgPsiCaClientToServer(val encryptedLabels: List<BigInteger>, val client: Int) : Serializable {
    override fun serialize(): ByteArray {
        return ByteArrayOutputStream().use { bos ->
            ObjectOutputStream(bos).use { oos ->
                oos.writeObject(encryptedLabels)
                oos.writeInt(client)
                oos.flush()
            }
            bos
        }.toByteArray()
    }

    companion object Deserializer : Deserializable<MsgPsiCaClientToServer> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgPsiCaClientToServer, Int> {
            val croppedBuffer = buffer.copyOfRange(offset, buffer.size)
            ByteArrayInputStream(croppedBuffer).use { bis ->
                ObjectInputStream(bis).use { ois ->
                    return Pair(MsgPsiCaClientToServer(ois.readObject() as List<BigInteger>, ois.readInt()), buffer.size)
                }
            }
        }
    }
}

data class MsgPsiCaServerToClient(
    val reEncryptedLabels: List<BigInteger>,
    @Suppress("UnstableApiUsage") val bloomFilter: BloomFilter<BigInteger>,
    val server: Int,
) : Serializable {
    override fun serialize(): ByteArray {
        return ByteArrayOutputStream().use { bos ->
            ObjectOutputStream(bos).use { oos ->
                oos.writeObject(reEncryptedLabels)
                oos.writeObject(bloomFilter)
                oos.writeInt(server)
                oos.flush()
            }
            bos
        }.toByteArray()
    }

    companion object Deserializer : Deserializable<MsgPsiCaServerToClient> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgPsiCaServerToClient, Int> {
            val croppedBuffer = buffer.copyOfRange(offset, buffer.size)
            ByteArrayInputStream(croppedBuffer).use { bis ->
                ObjectInputStream(bis).use { ois ->
                    return Pair(MsgPsiCaServerToClient(ois.readObject() as List<BigInteger>,
                        ois.readObject() as BloomFilter<BigInteger>,
                        ois.readInt()), buffer.size)
                }
            }
        }
    }
}

data class MsgNotifyHeartbeat(val unused: Boolean) : Serializable {
    override fun serialize(): ByteArray {
        return byteArrayOf()
    }

    companion object Deserializer : Deserializable<MsgNotifyHeartbeat> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgNotifyHeartbeat, Int> {
            // Unused
            throw RuntimeException("Only to be used by the master, not the slave")
        }
    }
}

data class MsgNewTestCommand(val configuration: Map<String, String>) : Serializable {
    val parsedConfiguration: MLConfiguration

    init {
        val iteratorDistribution_ = configuration.getValue("iteratorDistribution")
        val iteratorDistribution = if (iteratorDistribution_.startsWith('[')) {
            iteratorDistribution_
                .substring(1, iteratorDistribution_.length - 1)
                .split(", ")
                .map { it.toInt() }
        } else {
            loadIteratorDistribution(iteratorDistribution_)!!.value
        }
        parsedConfiguration = MLConfiguration(
            loadDataset(configuration.getValue("dataset"))!!,
            DatasetIteratorConfiguration(
                batchSize = loadBatchSize(configuration.getValue("batchSize"))!!,
                maxTestSamples = loadMaxTestSample(configuration.getValue("maxTestSamples"))!!,
                distribution = iteratorDistribution
            ),
            NNConfiguration(
                optimizer = loadOptimizer(configuration.getValue("optimizer"))!!,
                learningRate = loadLearningRate(configuration.getValue("learningRate"))!!,
                momentum = loadMomentum(configuration.getValue("momentum"))!!,
                l2 = loadL2Regularization(configuration.getValue("l2"))!!
            ),
            TrainConfiguration(
                maxIteration = loadMaxIteration(configuration.getValue("maxIterations"))!!,
                gar = loadGAR(configuration.getValue("gar"))!!,
                communicationPattern = loadCommunicationPattern(configuration.getValue("communicationPattern"))!!,
                behavior = loadBehavior(configuration.getValue("behavior"))!!,
                slowdown = loadSlowdown(configuration.getValue("slowdown"))!!,
                joiningLate = loadTransmissionRound(configuration.getValue("joiningLate"))!!
            ),
            ModelPoisoningConfiguration(
                attack = loadModelPoisoningAttack(configuration.getValue("modelPoisoningAttack"))!!,
                numAttackers = loadNumAttackers(configuration.getValue("numAttackers"))!!
            )
        )
    }

    override fun serialize(): ByteArray {
        // Unused
        throw RuntimeException("Only to be used by the master, not the slave")
    }

    companion object Deserializer : Deserializable<MsgNewTestCommand> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgNewTestCommand, Int> {
            val croppedBuffer = buffer.copyOfRange(offset, buffer.size)
            ByteArrayInputStream(croppedBuffer).use { bis ->
                ObjectInputStream(bis).use { ois ->
                    return Pair(MsgNewTestCommand(ois.readObject() as Map<String, String>), buffer.size)
                }
            }
        }
    }
}

data class MsgNotifyEvaluation(val evaluation: String) : Serializable {
    override fun serialize(): ByteArray {
        return ByteArrayOutputStream().use { bos ->
            ObjectOutputStream(bos).use { oos ->
                oos.writeObject(evaluation)
                oos.flush()
            }
            bos
        }.toByteArray()
    }

    companion object Deserializer : Deserializable<MsgNotifyEvaluation> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgNotifyEvaluation, Int> {
            // Unused
            throw RuntimeException("Only to be used by the master, not the slave")
        }
    }
}

data class MsgNotifyFinished(val unused: Boolean) : Serializable {
    override fun serialize(): ByteArray {
        return byteArrayOf()
    }

    companion object Deserializer : Deserializable<MsgNotifyFinished> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgNotifyFinished, Int> {
            // Unused
            throw RuntimeException("Only to be used by the master, not the slave")
        }
    }
}
