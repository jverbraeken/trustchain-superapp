package nl.tudelft.trustchain.fedml.ipv8

import android.util.Log
//import mu.KotlinLogging
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

    class Factory(
        private val settings: TrustChainSettings,
        private val database: TrustChainStore,
        private val crawler: TrustChainCrawler = TrustChainCrawler()
    ) : Overlay.Factory<FedMLCommunity>(FedMLCommunity::class.java) {
        override fun create(): FedMLCommunity {
            return FedMLCommunity(settings, database, crawler)
        }
    }

    // I'm claiming range 100 - 120
    // TODO: we should really create a class for the whole superapp project that manages the generation of unique message IDs
    enum class MessageId(val id: Int) {
        MSG_PING(100),
        MSG_PONG(101),
        MSG_PARAM_UPDATE(102)
    }

    init {
        messageHandlers[MessageId.MSG_PING.id] = ::onMsgPing
        messageHandlers[MessageId.MSG_PONG.id] = ::onMsgPong
        messageHandlers[MessageId.MSG_PARAM_UPDATE.id] = ::onMsgParamUpdate
        messageListeners[MessageId.MSG_PING]!!.add(object : MessageListener {
            override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
                sendToPeer(peer, MessageId.MSG_PONG, MsgPong("Pong"))
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
    internal fun sendToPeer(peer: Peer, messageID: MessageId, message: Serializable, logging: Boolean = false) {
        val packet = serializePacket(messageID.id, message, true, logging = logging)
        send(peer, packet)
    }

    internal fun sendToAll(messageID: MessageId, message: Serializable, logging: Boolean = false) {
        for (peer in getPeers()) {
            sendToPeer(peer, messageID, message, logging)
        }
    }

    internal fun sendToRandomPeer(messageID: MessageId, message: Serializable, logging: Boolean = false) {
        val peer = getPeers().random()
        sendToPeer(peer, messageID, message, logging)
    }

    // Round Robin
    internal fun sendToNextPeerRR(messageID: MessageId, message: Serializable, logging: Boolean = false) {
        val peer = getAndSetNextPeerRR()
        sendToPeer(peer, messageID, message, logging)
    }

    private fun getAndSetNextPeerRR(): Peer {
        if (peersRR?.isEmpty() != false) {
            peersRR = getPeers().toMutableList()
        }
        return peersRR!!.removeAt(0)
    }

    ////// MESSAGE RECEIVED EVENTS

    private fun onMsgPing(packet: Packet) {
        val (peer, payload) = packet.getAuthPayload(MsgPing.Deserializer)
        Log.e("MsgPing", peer.mid + ": " + payload.message)
        messageListeners[MessageId.MSG_PING]!!.forEach { it.onMessageReceived(MessageId.MSG_PING, peer, payload) }
    }

    private fun onMsgPong(packet: Packet) {
        val (peer, payload) = packet.getAuthPayload(MsgPong.Deserializer)
        Log.e("MsgPong", peer.mid + ": " + payload.message)
        messageListeners[MessageId.MSG_PONG]!!.forEach { it.onMessageReceived(MessageId.MSG_PONG, peer, payload) }
    }

    private fun onMsgParamUpdate(packet: Packet) {
        val (peer, payload) = packet.getAuthPayload(MsgParamUpdate.Deserializer)
        Log.e(" MsgParamUpdate", peer.mid)
        messageListeners[MessageId.MSG_PARAM_UPDATE]!!.forEach { it.onMessageReceived(MessageId.MSG_PARAM_UPDATE, peer, payload) }
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

data class MsgParamUpdate(val array: INDArray, val weight: Int) : Serializable {
    override fun serialize(): ByteArray {
        return ByteArrayOutputStream().use { ObjectOutputStream(it).writeObject(array); it }.toByteArray()
        /*val bos = ByteArrayOutputStream()
        ObjectOutputStream(bos).use {
            it.writeObject(array)
//            it.writeInt(weight)
            it.flush()
            return bos.toByteArray()
        }*/
    }

    companion object Deserializer : Deserializable<MsgParamUpdate> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgParamUpdate, Int> {
            val bis = ByteArrayInputStream(buffer.copyOfRange(offset, buffer.size))
            val ois = ObjectInputStream(bis)
            return Pair(MsgParamUpdate(ois.readObject() as INDArray, 9999), buffer.size)
            /*ObjectInputStream(bis).use {
                return Pair(MsgParamUpdate(it.readObject() as INDArray, 1812*//*it.readInt()*//*), buffer.size)
            }*/
        }
    }
}
