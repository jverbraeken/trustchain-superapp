package nl.tudelft.trustchain.fedml.ipv8

import com.google.common.hash.BloomFilter
import nl.tudelft.ipv8.messaging.Deserializable
import nl.tudelft.ipv8.messaging.Serializable
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.*
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import java.math.BigInteger

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

data class MsgParamUpdate(val array: INDArray, val iteration: Int) : Serializable {
    override fun serialize(): ByteArray {
        return ByteArrayOutputStream().use { bos ->
            ObjectOutputStream(bos).use { oos ->
                oos.writeObject(array);
                oos.writeInt(iteration)
                oos.flush()
            }
            bos
        }.toByteArray()
    }

    companion object Deserializer : Deserializable<MsgParamUpdate> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgParamUpdate, Int> {
            val croppedBuffer = buffer.copyOfRange(offset, buffer.size)
            ByteArrayInputStream(croppedBuffer).use { bis ->
                ObjectInputStream(bis).use { ois ->
                    return Pair(MsgParamUpdate(ois.readObject() as INDArray, ois.readInt()), buffer.size)
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
                    return Pair(
                        MsgPsiCaServerToClient(
                            ois.readObject() as List<BigInteger>,
                            ois.readObject() as BloomFilter<BigInteger>,
                            ois.readInt()
                        ), buffer.size
                    )
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

data class MsgNewTestCommand(val configuration: Map<String, String>, val figureName: String) : Serializable {
    val parsedConfiguration: MLConfiguration

    init {
        val iteratorDistribution_ = configuration.getValue("iteratorDistribution")
        val iteratorDistribution = if (iteratorDistribution_.startsWith('[')) {
            iteratorDistribution_
                .substring(1, iteratorDistribution_.length - 1)
                .split(", ")
                .map { it.toInt() }
        } else {
            loadIteratorDistribution(iteratorDistribution_)!!.value.toList()
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
                joiningLate = loadTransmissionRound(configuration.getValue("joiningLate"))!!,
                iterationsBeforeEvaluation = configuration.getValue("iterationsBeforeEvaluation").toInt(),
                iterationsBeforeSending = configuration.getValue("iterationsBeforeSending").toInt(),
                transfer = configuration.getValue("transfer").toBoolean()
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
                    return Pair(MsgNewTestCommand(ois.readObject() as Map<String, String>, ois.readObject() as String), buffer.size)
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

data class MsgForcedIntroduction(
    val wanPorts: List<Int>,
    val supportsTFTP: Boolean,
    val supportsUTP: Boolean,
    val serviceId: String
) : Serializable {
    override fun serialize(): ByteArray {
        // Unused
        throw RuntimeException("Only to be used by the master, not the slave")
    }

    companion object Deserializer : Deserializable<MsgForcedIntroduction> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MsgForcedIntroduction, Int> {
            val croppedBuffer = buffer.copyOfRange(offset, buffer.size)
            ByteArrayInputStream(croppedBuffer).use { bis ->
                ObjectInputStream(bis).use { ois ->
                    return Pair(
                        MsgForcedIntroduction(
                            ois.readObject() as List<Int>,
                            ois.readBoolean(),
                            ois.readBoolean(),
                            ois.readObject() as String
                        ),
                        buffer.size
                    )
                }
            }
        }
    }
}
