package nl.tudelft.trustchain.fedml.ipv8

import com.google.common.hash.BloomFilter
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
import nl.tudelft.ipv8.messaging.tftp.TFTPCommunity
import nl.tudelft.ipv8.messaging.utp.UTPCommunity
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.*
import org.apache.commons.net.tftp.TFTPAckPacket
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
        MSG_NOTIFY_FINISHED(113, MsgNotifyFinished.Deserializer),
        MSG_FORCED_INTRODUCTION(114, MsgForcedIntroduction.Deserializer),
    }

    init {
        messageHandlers[MessageId.MSG_PING.id] = ::onMsgPing
        messageHandlers[MessageId.MSG_PONG.id] = ::onMsgPong
        messageHandlers[MessageId.MSG_PARAM_UPDATE.id] = ::onMsgParamUpdate
        messageHandlers[MessageId.MSG_PSI_CA_CLIENT_TO_SERVER.id] = ::onMsgPsiCaClientToServer
        messageHandlers[MessageId.MSG_PSI_CA_SERVER_TO_CLIENT.id] = ::onMsgPsiCaServerToClient
        messageHandlers[MessageId.MSG_NEW_TEST_COMMAND.id] = ::onMsgNewTestCommand
        messageHandlers[MessageId.MSG_FORCED_INTRODUCTION.id] = ::onMsgForcedIntroduction

        messageListeners[MessageId.MSG_PING]!!.add(object : MessageListener {
            override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
                sendToPeer(peer, MessageId.MSG_PONG, MsgPong("Pong"), reliable = true)
            }
        })
        messageListeners[MessageId.MSG_FORCED_INTRODUCTION]!!.add(object : MessageListener {
            override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
                logger.debug { "Received MSG_FORCED_INTRODUCTION" }
                network.removeAllPeers()
                val msgForcedIntroduction = payload as MsgForcedIntroduction
                val wan = network.wanLog.estimateWan()!!
                for (wanPort in msgForcedIntroduction.wanPorts) {
                    logger.debug { "1: ${getPeers().size}"}
                    logger.debug { "wanPort: $wanPort" }
                    val address = IPv4Address(wan.ip, wanPort)
                    val introPeer = Peer(
                        defaultCryptoProvider.generateKey(),
                        address,
                        address,
                        address,
                        supportsTFTP = msgForcedIntroduction.supportsTFTP,
                        supportsUTP = msgForcedIntroduction.supportsUTP,
                    )
                    network.addVerifiedPeer(introPeer)
                    logger.debug { "2: ${getPeers().size}"}
                    val services = arrayListOf(msgForcedIntroduction.serviceId)
                    if (msgForcedIntroduction.supportsTFTP) services.add(TFTPCommunity.SERVICE_ID)
                    if (msgForcedIntroduction.supportsUTP) services.add(UTPCommunity.SERVICE_ID)
                    logger.debug { "3: ${getPeers().size}"}
                    network.discoverServices(introPeer, services)
                    logger.debug { "4: ${getPeers().size}"}
                    logger.debug { "discovered services" }
                }
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
        writeWanAddress(baseDirectory)
        keepSendingHeartbeats()
    }

    private fun writeWanAddress(baseDirectory: File) = thread {
        while (myEstimatedWan.port == 0) {
            Thread.sleep(100)
        }
        val file = File(baseDirectory, "wanPort")
        file.delete()
        file.createNewFile()
        PrintWriter(file).use {
            it.println(myEstimatedWan.ip)
            it.println(myEstimatedWan.port)
        }
    }

    private fun keepSendingHeartbeats() = thread {
        while (true) {
            sendToMaster(MessageId.MSG_NOTIFY_HEARTBEAT, MsgNotifyHeartbeat(true))
            Thread.sleep(2000)
        }
    }

    internal fun sendToMaster(messageID: MessageId, message: Serializable, logging: Boolean = false, reliable: Boolean = false) {
//        logger.debug { "sendToMaster, messageId: ${messageID.id}" }
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
        logger.debug { "sendToPeer: ${peer.address.port}, messageId: ${messageID.id}" }
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
            logger.debug { "SendToAll peer: ${peer.address.port}, ${messageID.id}" }
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

    private fun onMsgForcedIntroduction(packet: Packet) {
        onMessage(packet, MessageId.MSG_FORCED_INTRODUCTION)
    }

    private fun onMessage(packet: Packet, messageId: MessageId) {
        val (peer, payload) = packet.getAuthPayload(messageId.deserializer)
        logger.debug { "${messageId.name}: ${peer.mid}" }
        messageListeners[messageId]!!.forEach { it.onMessageReceived(messageId, peer, payload) }
    }
}
