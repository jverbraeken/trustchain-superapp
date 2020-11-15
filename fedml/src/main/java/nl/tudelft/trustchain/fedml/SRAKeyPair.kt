package nl.tudelft.trustchain.fedml

import java.math.BigInteger
import java.util.*


class SRAKeyPair private constructor(
    private val prime: BigInteger,
    private val secret: BigInteger,
    private val secretInverse: BigInteger
) {
    fun encrypt(message: BigInteger): BigInteger {
        return message.modPow(secret, prime)
    }

    fun decrypt(cypher: BigInteger): BigInteger {
        return cypher.modPow(secretInverse, prime)
    }

    companion object {
        private const val DEFAULT_NUM_BITS = 32

        fun create(prime: BigInteger, random: Random): SRAKeyPair {
            val secret = generateEncryptionKey(prime, DEFAULT_NUM_BITS, random)
            val secretInverse = secret.modInverse(prime.subtract(BigInteger.ONE))
            return SRAKeyPair(prime, secret, secretInverse)
        }

        private fun generateEncryptionKey(
            p: BigInteger,
            numBits: Int,
            random: Random
        ): BigInteger {
            val phiP = p.subtract(BigInteger.ONE)
            // Choose a key that is invertible in mod phi(prime)
            var k = BigInteger(numBits, random)
            while (k.gcd(phiP) != BigInteger.ONE) {
                k = BigInteger(numBits, random)
            }
            return k
        }
    }
}
