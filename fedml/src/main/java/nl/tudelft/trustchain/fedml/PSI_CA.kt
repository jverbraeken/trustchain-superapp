@file:Suppress("UnstableApiUsage")

package nl.tudelft.trustchain.fedml

import com.google.common.hash.BloomFilter
import com.google.common.hash.Funnel
import com.google.common.hash.PrimitiveSink
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaClientToServer
import nl.tudelft.trustchain.fedml.ipv8.MsgPsiCaServerToClient
import java.math.BigInteger
import java.util.concurrent.CopyOnWriteArrayList
import java.util.stream.Collectors

private const val MIN_PSI_CA = 3
private const val SIZE_BLOOM_FILTER = 1000
private val logger = KotlinLogging.logger("PSI_CA")

data class ToServerMessage(
    val encryptedLabels: List<BigInteger>,
    val client: Int
)

data class ToClientMessage1(
    val encryptedLabels: List<BigInteger>,
    val server: Int
)

data class ToClientMessage2(
    val bloomFilter: BloomFilter<BigInteger>,
    val server: Int
)

fun clientsRequestsServerLabels(
    labels: List<String>,
    sraKeyPair: SRAKeyPair
): ArrayList<BigInteger> {
    val encryptedLabels = ArrayList<BigInteger>(labels.size)
    for (label in labels) {
        val message = BigInteger.valueOf(label.hashCode().toLong())
        val encryptedLabel = sraKeyPair.encrypt(message)
        encryptedLabels.add(encryptedLabel)
    }
    return encryptedLabels
}

fun serverRespondsClientRequests(
    labels: List<String>,
    toServerMessage: MsgPsiCaClientToServer,
    sraKeyPair: SRAKeyPair
): Pair<List<BigInteger>, BloomFilter<BigInteger>> {
    val encryptedLabels = ArrayList<BigInteger>(labels.size)
    for (label in labels) {
        val message = BigInteger.valueOf(label.hashCode().toLong())
        val encryptedLabel = sraKeyPair.encrypt(message)
        encryptedLabels.add(encryptedLabel)
    }

    val filter = BloomFilter.create(BigIntegerFunnel(), SIZE_BLOOM_FILTER)
    encryptedLabels.forEach { filter.put(it) }

    val shuffledLabels = toServerMessage.encryptedLabels.shuffled()
    val reEncryptedLabels = shuffledLabels.stream().map { sraKeyPair.encrypt(it) }.collect(Collectors.toList())

    return Pair(reEncryptedLabels, filter)
}

fun clientReceivesServerResponses(
    i: Int,
    toClientMessageBuffers: CopyOnWriteArrayList<MsgPsiCaServerToClient>,
    sraKeyPair: SRAKeyPair
): List<Int> {
    val similarPeers = ArrayList<Int>()
    for (buffer1 in toClientMessageBuffers) {
        val semiDecryptedLabels = buffer1.reEncryptedLabels.map { sraKeyPair.decrypt(it) }
        val bloomFilter = toClientMessageBuffers.first { it.server == buffer1.server }.bloomFilter
        val count = semiDecryptedLabels.filter { bloomFilter.mightContain(it) }.size
        if (count >= MIN_PSI_CA) {
            similarPeers.add(buffer1.server)
            logger.debug { "Peer $i will send to peer ${buffer1.server}: overlap = $count" }
        }
    }
    return similarPeers
}

private class BigIntegerFunnel : Funnel<BigInteger> {
    override fun funnel(from: BigInteger, into: PrimitiveSink) {
        into.putBytes(from.toByteArray())
    }
}
