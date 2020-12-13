package nl.tudelft.trustchain.fedml.ui

import kotlinx.serialization.Serializable

@Serializable
data class Automation(val fixedValues: Map<String, String>, val figures: List<Figure>)

@Serializable
data class Figure(val name: String, val fixedValues: Map<String, String>, val tests: List<Test>, val iteratorDistributions: List<List<Int>>? = null)

@Serializable
data class Test(val gar: String)
