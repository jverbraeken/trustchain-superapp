package nl.tudelft.trustchain.fedml.ipv8

import android.util.Log
import nl.tudelft.ipv8.Overlay
import nl.tudelft.ipv8.Peer
import nl.tudelft.ipv8.attestation.trustchain.TrustChainCommunity
import nl.tudelft.ipv8.attestation.trustchain.TrustChainCrawler
import nl.tudelft.ipv8.attestation.trustchain.TrustChainSettings
import nl.tudelft.ipv8.attestation.trustchain.store.TrustChainStore
import nl.tudelft.ipv8.messaging.Deserializable
import nl.tudelft.ipv8.messaging.Packet
import nl.tudelft.ipv8.messaging.Serializable
import org.nd4j.linalg.cpu.nativecpu.NDArray
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream

class FedMLCommunity(
    settings: TrustChainSettings,
    database: TrustChainStore,
    crawler: TrustChainCrawler = TrustChainCrawler()
) : TrustChainCommunity(settings, database, crawler) {
    override val serviceId = "29384902d2938f34872398758cf7ca9238ccc333"

    class Factory(
        private val settings: TrustChainSettings,
        private val database: TrustChainStore,
        private val crawler: TrustChainCrawler = TrustChainCrawler()
    ) : Overlay.Factory<FedMLCommunity>(
        FedMLCommunity::class.java
    ) {
        override fun create(): FedMLCommunity {
            return FedMLCommunity(
                settings,
                database,
                crawler
            )
        }
    }

    object MessageId {
        const val MSG_PING = 0
        const val MSG_PONG = 1
        const val MSG_PARAM_UPDATE = 2
    }

    internal fun sendToPeer(peer: Peer, messageID: Int, message: Serializable) {
        val packet = serializePacket(messageID, message, true)
        send(peer.address, packet)
    }

    internal fun sendToAll(messageID: Int, message: Serializable) {
        for (peer in getPeers()) {
            sendToPeer(peer, messageID, message)
        }
    }

    init {
        messageHandlers[MessageId.MSG_PING] = ::onMsgPing
        messageHandlers[MessageId.MSG_PONG] = ::onMsgPong
        messageHandlers[MessageId.MSG_PARAM_UPDATE] = ::onMsgParamUpdate
    }

    private fun onMsgPing(packet: Packet) {
        val (peer, payload) = packet.getAuthPayload(MsgPing.Deserializer)
        Log.e("MsgPing", peer.mid + ": " + payload.message)
        sendToAll(MessageId.MSG_PONG, MsgPong("Pong"))
    }

    private fun onMsgPong(packet: Packet) {
        val (peer, payload) = packet.getAuthPayload(MsgPong.Deserializer)
        Log.e("MsgPong", peer.mid + ": " + payload.message)
    }

    private fun onMsgParamUpdate(packet: Packet) {
        val (peer, _) = packet.getAuthPayload(MsgParamUpdate.Deserializer)
        Log.e("MsgParamUpdate", peer.mid)
    }
}


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

data class MsgParamUpdate(val message: NDArray) : Serializable {
    override fun serialize(): ByteArray {
        val bos = ByteArrayOutputStream()
        ObjectOutputStream(bos).use {
            it.writeObject(message)
            it.flush()
            return bos.toByteArray()
        }
    }

    companion object Deserializer : Deserializable<MsgParamUpdate> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgParamUpdate, Int> {
            val bis = ByteArrayInputStream(buffer)
            ObjectInputStream(bis).use {
                return Pair(it.readObject() as MsgParamUpdate, buffer.size)
            }
        }
    }
}
