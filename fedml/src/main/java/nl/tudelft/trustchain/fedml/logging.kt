package nl.tudelft.trustchain.fedml

import mu.KLogger

    fun KLogger.d(logging: Boolean, msg: () -> String) {
        if (logging) {
            debug { msg.invoke() }
        }
    }

    fun KLogger.t(logging: Boolean, msg: () -> String) {
        if (logging) {
            trace { msg.invoke() }
        }
    }
