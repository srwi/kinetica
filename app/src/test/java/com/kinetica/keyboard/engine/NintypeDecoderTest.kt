package com.kinetica.keyboard.engine

import com.kinetica.keyboard.engine.models.StreamId
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NintypeDecoderTest {

    private val g = TestData.qwertyGeometry()

    private fun testTrie(): Trie = Trie.build(
        listOf(
            "the" to 100_000,
            "to" to 90_000,
            "too" to 50_000,
            "winter" to 20_000,
            "when" to 30_000,
            "cuando" to 80_000,
            "cunado" to 5_000,
            "praticamente" to 15_000,
            "pratica" to 10_000,
            "wanted" to 25_000,
        )
    )

    @Test
    fun testFastCurveMapperExactPoints() {
        val cov = NintypeDecoder.FastCurveMapper.createCoverageCurve()
        assertEquals(1.00f, cov.get(0.00f), 1e-4f)
        assertEquals(1.00f, cov.get(0.25f), 1e-4f)
        assertEquals(0.80f, cov.get(0.50f), 1e-4f)
        assertEquals(0.40f, cov.get(0.75f), 1e-4f)
        assertEquals(0.25f, cov.get(1.00f), 1e-4f)
        assertEquals(0.10f, cov.get(1.25f), 1e-4f)
        assertEquals(0.00f, cov.get(1.50f), 1e-4f)
        assertEquals(0.00f, cov.get(2.00f), 1e-4f)

        val skip = NintypeDecoder.FastCurveMapper.createSkipPenaltyCurve()
        assertEquals(1.00f, skip.get(0.00f), 1e-4f)
        assertEquals(0.85f, skip.get(0.10f), 1e-4f)
        assertEquals(0.75f, skip.get(0.143f), 1e-4f)
        assertEquals(0.60f, skip.get(0.167f), 1e-4f)
        assertEquals(0.40f, skip.get(0.20f), 1e-4f)
        assertEquals(0.20f, skip.get(0.33f), 1e-4f)
        assertEquals(0.05f, skip.get(0.50f), 1e-4f)
        assertEquals(0.00f, skip.get(0.60f), 1e-4f)

        val ppsck = NintypeDecoder.FastCurveMapper.createPPSCKCurve()
        assertEquals(1.00f, ppsck.get(500.0f), 1e-4f)
        assertEquals(0.50f, ppsck.get(1000.0f), 1e-4f)
        assertEquals(0.10f, ppsck.get(1400.0f), 1e-4f)
        assertEquals(0.00f, ppsck.get(2000.0f), 1e-4f)
    }

    @Test
    fun testSingleSwipeDecodesThe() {
        val trie = testTrie()
        val predictor = WordPredictor(trie, BigramTable.EMPTY, g)
        val decoder = NintypeDecoder(predictor, trie, BigramTable.EMPTY, g, -1, null)

        val tokens = listOf(
            TestData.swipe("the", g, t0 = 0, durMs = 300, stream = StreamId.LEFT)
        )

        val candidates = decoder.decode(tokens)
        assertTrue("Should return candidates", candidates.isNotEmpty())
        val words = candidates.map { it.word }
        assertTrue("Expected 'the' in candidates, got $words", words.contains("the"))
        assertEquals("Expected 'the' as top-1 candidate", "the", words.first())
    }

    @Test
    fun testDualSimultaneousSwipeDecodesWinter() {
        val trie = testTrie()
        val predictor = WordPredictor(trie, BigramTable.EMPTY, g)
        val decoder = NintypeDecoder(predictor, trie, BigramTable.EMPTY, g, -1, null)

        // Simultaneous / partially overlapping swipes:
        // Left finger: w-i-e-r (0 to 600 ms)
        // Right finger: n-t (200 to 500 ms)
        val tokens = listOf(
            TestData.swipe("wier", g, t0 = 0, durMs = 600, stream = StreamId.LEFT),
            TestData.swipe("nt", g, t0 = 200, durMs = 300, stream = StreamId.RIGHT)
        )

        val candidates = decoder.decode(tokens)
        assertTrue("Should return candidates for dual swipe", candidates.isNotEmpty())
        val words = candidates.map { it.word }
        assertTrue("Expected 'winter' in candidates for dual swipe, got $words", words.contains("winter"))
    }

    @Test
    fun testInterleavedSwipeAndTapDecodesCuando() {
        val trie = testTrie()
        val predictor = WordPredictor(trie, BigramTable.EMPTY, g)
        val decoder = NintypeDecoder(predictor, trie, BigramTable.EMPTY, g, -1, null)

        // Left swipe c-u-n-o (0 to 900 ms)
        // Right tap 'a' at 300 ms, 'd' at 600 ms
        val tokens = listOf(
            TestData.swipe("cuno", g, t0 = 0, durMs = 900, stream = StreamId.LEFT),
            TestData.tap('a', g, 300, StreamId.RIGHT),
            TestData.tap('d', g, 600, StreamId.RIGHT)
        )

        val candidates = decoder.decode(tokens)
        assertTrue("Should return candidates for interleaved swipe and taps", candidates.isNotEmpty())
        val words = candidates.map { it.word }
        assertTrue("Expected 'cuando' in candidates, got $words", words.contains("cuando"))
    }

    @Test
    fun testWordPredictorEndToEndDualSimultaneousSwipe() {
        val trie = testTrie()
        val predictor = WordPredictor(trie, BigramTable.EMPTY, g)

        val tokens = listOf(
            TestData.swipe("wier", g, t0 = 0, durMs = 600, stream = StreamId.LEFT),
            TestData.swipe("nt", g, t0 = 200, durMs = 300, stream = StreamId.RIGHT)
        )

        val candidates = predictor.decode(tokens, emptyList())
        val words = candidates.map { it.word }
        assertTrue("Expected 'winter' in end-to-end predictor decode, got $words", "winter" in words)
    }

    @Test
    fun testWordPredictorEndToEndSingleSwipe() {
        val trie = testTrie()
        val predictor = WordPredictor(trie, BigramTable.EMPTY, g)

        val tokens = listOf(
            TestData.swipe("the", g, t0 = 0, durMs = 300, stream = StreamId.LEFT)
        )

        val candidates = predictor.decode(tokens, emptyList())
        val words = candidates.map { it.word }
        assertTrue("Expected 'the' in end-to-end predictor decode, got $words", "the" in words)
        assertEquals("the", words.first())
    }

    @Test
    fun testWordPredictorToggleNintypeEnabled() {
        val trie = testTrie()
        val predictor = WordPredictor(trie, BigramTable.EMPTY, g, nintypeEnabled = true)
        assertTrue(predictor.nintypeEnabled)

        // Non-chronological dual swipe: Left finger swipes "wn" (0..200ms), Right finger swipes "iter" (250..750ms).
        // Strict chronological ordering is "w-n-i-t-e-r", which fails Kinetica's native DFS.
        val dualSwipe = listOf(
            TestData.swipe("wn", g, t0 = 0, durMs = 200, stream = StreamId.LEFT),
            TestData.swipe("iter", g, t0 = 250, durMs = 500, stream = StreamId.RIGHT)
        )

        // When ON: Nintype multi-finger anchors decode "winter"
        val onCandidates = predictor.decode(dualSwipe, emptyList())
        val onWords = onCandidates.map { it.word }
        assertTrue("Expected 'winter' in onCandidates, got $onWords", "winter" in onWords)

        // When OFF: Kinetica original engine only, cannot decode non-chronological "wn"+"iter" into "winter"
        predictor.nintypeEnabled = false
        assertFalse(predictor.nintypeEnabled)
        val offCandidates = predictor.decode(dualSwipe, emptyList())
        val offWords = offCandidates.map { it.word }
        assertFalse("Did not expect 'winter' in offWords, got $offWords", "winter" in offWords)
    }
}
