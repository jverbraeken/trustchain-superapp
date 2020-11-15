package nl.tudelft.trustchain.fedml.ipv8

import com.google.common.hash.BloomFilter
import mu.KotlinLogging
import nl.tudelft.ipv8.Overlay
import nl.tudelft.ipv8.Peer
import nl.tudelft.ipv8.attestation.trustchain.TrustChainCommunity
import nl.tudelft.ipv8.attestation.trustchain.TrustChainCrawler
import nl.tudelft.ipv8.attestation.trustchain.TrustChainSettings
import nl.tudelft.ipv8.attestation.trustchain.store.TrustChainStore
import nl.tudelft.ipv8.messaging.Deserializable
import nl.tudelft.ipv8.messaging.Packet
import nl.tudelft.ipv8.messaging.Serializable
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import java.math.BigInteger

private val logger = KotlinLogging.logger("FedMLCommunity")

interface MessageListener {
    fun onMessageReceived(messageId: FedMLCommunity.MessageId, peer: Peer, payload: Any)
}


class FedMLCommunity(
    settings: TrustChainSettings,
    database: TrustChainStore,
    crawler: TrustChainCrawler = TrustChainCrawler()
) : TrustChainCommunity(settings, database, crawler) {
    override val serviceId = "36b098237ff4debfd0278b8b87c583e1c2cce4b7"
    private var peersRR: MutableList<Peer>? = null
    private var peersRing: MutableList<Peer>? = null
    private var ringCounter: Int = 1

    class Factory(
        private val settings: TrustChainSettings,
        private val database: TrustChainStore,
        private val crawler: TrustChainCrawler = TrustChainCrawler()
    ) : Overlay.Factory<FedMLCommunity>(FedMLCommunity::class.java) {
        override fun create(): FedMLCommunity {
            return FedMLCommunity(settings, database, crawler)
        }
    }

    override fun onPacket(packet: Packet) {
        super.onPacket(packet)
    }

    // I'm claiming range 100 - 120
    // TODO: we should really create a class for the whole superapp project that manages the generation of unique message IDs
    enum class MessageId(val id: Int, val deserializer: Deserializable<out Any>) {
        MSG_PING(100, MsgPing.Deserializer),
        MSG_PONG(101, MsgPong.Deserializer),
        MSG_PARAM_UPDATE(102, MsgParamUpdate.Deserializer),
        MSG_PSI_CA_CLIENT_TO_SERVER(103, MsgPsiCaClientToServer.Deserializer),
        MSG_PSI_CA_SERVER_TO_CLIENT(104, MsgPsiCaServerToClient.Deserializer)
    }

    init {
        messageHandlers[MessageId.MSG_PING.id] = ::onMsgPing
        messageHandlers[MessageId.MSG_PONG.id] = ::onMsgPong
        messageHandlers[MessageId.MSG_PARAM_UPDATE.id] = ::onMsgParamUpdate
        messageHandlers[MessageId.MSG_PSI_CA_CLIENT_TO_SERVER.id] = ::onMsgPsiCaClientToServer
        messageHandlers[MessageId.MSG_PSI_CA_SERVER_TO_CLIENT.id] = ::onMsgPsiCaServerToClient

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

    @Suppress("MemberVisibilityCanBePrivate")
    internal fun sendToPeer(peer: Peer, messageID: MessageId, message: Serializable, logging: Boolean = false, reliable: Boolean = false) {
        logger.debug { "sendToPeer, messageId: ${messageID.id}"}
        val packet = serializePacket(messageID.id, message, true, logging = logging)
        send(peer, packet, reliable)
    }

    internal fun sendToAll(messageID: MessageId, message: Serializable, priorityPeers: List<Int>? = null, reliable: Boolean = false) {
        logger.debug { "sendToAll" }
        for (peer in getAllowedPeers(priorityPeers)) {
            logger.debug { "Peer: ${peer.address}" }
            sendToPeer(peer, messageID, message, reliable = reliable)
        }
    }

    internal fun sendToRandomPeer(messageID: MessageId, message: Serializable, priorityPeers: List<Int>? = null, reliable: Boolean = false) {
        logger.debug { "sendToRandomPeer" }
        val peers = getAllowedPeers(priorityPeers)
        if (peers.isNotEmpty()) {
            val peer = peers.random()
            logger.debug { "Peer: ${peer.address}" }
            sendToPeer(peer, messageID, message, reliable = reliable)
        }
    }

    private fun getAllowedPeers(peers: List<Int>?): List<Peer> {
        return getPeers().filter { if (peers == null) true else it.address.port in peers }
    }

    // Round Robin
    internal fun sendToNextPeerRR(messageID: MessageId, message: Serializable, priorityPeers: List<Int>? = null, reliable: Boolean = false) {
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

    private fun getAndSetNextPeerRR(priorityPeers: List<Int>? = null): Peer? {
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

    internal fun sendToNextPeerRing(messageID: MessageId, message: Serializable, priorityPeers: List<Int>? = null, reliable: Boolean = false) {
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

    private fun getAndSetNextPeerRing(priorityPeers: List<Int>? = null): Peer? {
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

data class MsgPsiCaServerToClient(val reEncryptedLabels: List<BigInteger>, @Suppress("UnstableApiUsage") val bloomFilter: BloomFilter<BigInteger>, val server: Int) : Serializable {
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
                    return Pair(MsgPsiCaServerToClient(ois.readObject() as List<BigInteger>, ois.readObject() as BloomFilter<BigInteger>, ois.readInt()), buffer.size)
                }
            }
        }
    }
}
