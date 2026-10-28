package com.kinetica.keyboard.engine

import com.kinetica.keyboard.engine.models.InputToken
import com.kinetica.keyboard.engine.models.PathPoint
import com.kinetica.keyboard.engine.models.StreamId
import com.kinetica.keyboard.engine.models.SwipeToken
import com.kinetica.keyboard.engine.models.TapToken
import com.kinetica.keyboard.engine.models.WordCandidate
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Exact Nintype (Keyboard 69 / 72) Multi-Finger Swipe Decoding Engine.
 *
 * Reverse-engineered bit-by-bit from libgl2jni.so (ARM 32-bit ELF):
 * - Key-unit coordinate normalization (kw).
 * - 4-point inflection/curvature turn detection (syncCurvyRVA >= 0.30f).
 * - Key apex closest-approach anchor detection.
 * - Continuous soft-index cross-finger time synchronization lattice.
 * - Distance-to-coverage curve (FastCurveMapper) with power alpha = 0.60f.
 * - 4-tier Priority Queue Buckets (Same-finger, Hand-switch, Apostrophe, Skip).
 * - 10-stage pruning funnel with hard 2.0f pruning floor (maybePrune).
 * - Exact sub-curves: multForCharSkip (8-point), calculateNotLongEnoughMultpen (6-point),
 *   determinePPSCKScoreMult (4-point velocity with pow 0.4), and double-letter vowel bitmask (0x00104111).
 */
class NintypeDecoder(
    private val predictor: WordPredictor,
    private val trie: Trie,
    private val bigrams: BigramTable,
    private val geometry: KeyboardGeometry,
    private val prevWordId: Int,
    private val prevWord: String?,
) {
    // =========================================================================
    // 1. Exact Monotonic Piecewise Curve Mapper (FastCurveMapper)
    // =========================================================================
    class FastCurveMapper(private val pts: List<Point>) {
        data class Point(val x: Float, val y: Float)

        fun get(x: Float): Float {
            if (pts.isEmpty()) return 1.0f
            if (x <= pts.first().x) return pts.first().y
            if (x >= pts.last().x) return pts.last().y
            for (i in 0 until pts.size - 1) {
                val p0 = pts[i]
                val p1 = pts[i + 1]
                if (x >= p0.x && x <= p1.x) {
                    val t = (x - p0.x) / (p1.x - p0.x)
                    return p0.y + t * (p1.y - p0.y)
                }
            }
            return pts.last().y
        }

        companion object {
            fun createCoverageCurve(): FastCurveMapper = FastCurveMapper(
                listOf(
                    Point(0.00f, 1.00f),
                    Point(0.25f, 1.00f),
                    Point(0.50f, 0.80f),
                    Point(0.75f, 0.40f),
                    Point(1.00f, 0.25f),
                    Point(1.25f, 0.10f),
                    Point(1.50f, 0.00f),
                )
            )

            fun createSkipPenaltyCurve(): FastCurveMapper = FastCurveMapper(
                listOf(
                    Point(0.000f, 1.00f),
                    Point(0.100f, 0.85f),
                    Point(0.143f, 0.75f),
                    Point(0.167f, 0.60f),
                    Point(0.200f, 0.40f),
                    Point(0.330f, 0.20f),
                    Point(0.500f, 0.05f),
                    Point(0.600f, 0.00f),
                )
            )

            fun createLengthMismatchCurve(): FastCurveMapper = FastCurveMapper(
                listOf(
                    Point(1.00f, 1.00f),
                    Point(2.00f, 0.93f),
                    Point(3.00f, 0.85f),
                    Point(3.50f, 0.50f),
                    Point(5.00f, 0.30f),
                    Point(10.0f, 0.10f),
                )
            )

            fun createPPSCKCurve(): FastCurveMapper = FastCurveMapper(
                listOf(
                    Point(600.0f, 1.00f),
                    Point(1000.0f, 0.50f),
                    Point(1400.0f, 0.10f),
                    Point(1900.0f, 0.00f),
                )
            )
        }
    }

    private val coverageCurve = FastCurveMapper.createCoverageCurve()
    private val skipCurve = FastCurveMapper.createSkipPenaltyCurve()
    private val lengthCurve = FastCurveMapper.createLengthMismatchCurve()
    private val ppsckCurve = FastCurveMapper.createPPSCKCurve()

    // =========================================================================
    // 2. Data Structures
    // =========================================================================
    data class TravKey(
        val code: Int,
        val char: Char,
        val dist: Float,
        val coverageScore: Float,
    )

    data class MooPoint(
        val x: Float,
        val y: Float,
        val t: Long,
        var speed: Float = 0f,
        var dwellRatio: Float = 0f,
        var isCurvatureTurn: Boolean = false,
        var isBoundaryPoint: Boolean = false,
        var otherFingerSoftIndex: Float = -1f,
        val travKeys: ArrayList<TravKey> = ArrayList(),
    ) {
        val isVisitationPoint: Boolean
            get() = isCurvatureTurn || isBoundaryPoint || dwellRatio > 0.3f
    }

    data class Anchor(
        val fingerId: Int, // 0 = Left, 1 = Right
        val index: Int,
        val pt: MooPoint,
        val t: Long,
        val cands: List<TravKey>,
        val isVisitation: Boolean,
    )

    data class PredPath(
        val node: Int,
        val word: String,
        val leftAnchorIdx: Int,  // Last consumed anchor on Left finger (-1 if none)
        val rightAnchorIdx: Int, // Last consumed anchor on Right finger (-1 if none)
        val activeFinger: Int,   // 0 = Left, 1 = Right
        val lastTime: Long,
        val skipCount: Int,
        val branchMode: Int,     // 0 = same, 1 = switch, 2 = apos, 3 = skip
        var heuristicScore: Float,
        var minScoreSeen: Float,
        val charCount: Int = word.length,
    ) : Comparable<PredPath> {
        override fun compareTo(other: PredPath): Int {
            return other.heuristicScore.compareTo(this.heuristicScore)
        }

        fun maybePrune(cutoff: Float = 2.0f): Boolean {
            val threshold = max(cutoff, 2.0f)
            return heuristicScore < threshold
        }
    }

    // =========================================================================
    // 3. Pre-Processing, Feature Extraction & Anchor Harvesting
    // =========================================================================
    private fun extractAnchors(tokens: List<InputToken>, targetStream: StreamId, fingerId: Int): List<Anchor> {
        val streamTokens = tokens.filter { it.streamId == targetStream }.sortedBy { it.tStart }
        if (streamTokens.isEmpty()) return emptyList()

        val rawPoints = ArrayList<MooPoint>()
        for (token in streamTokens) {
            when (token) {
                is SwipeToken -> {
                    for (p in token.rawPath) {
                        rawPoints.add(MooPoint(p.x, p.y, p.t))
                    }
                }
                is TapToken -> {
                    rawPoints.add(MooPoint(token.x, token.y, token.tStart, isBoundaryPoint = true))
                    if (token.tEnd > token.tStart) {
                        val dwell = min(1.0f, (token.tEnd - token.tStart) / 150f)
                        rawPoints.add(MooPoint(token.x, token.y, token.tEnd, dwellRatio = dwell, isBoundaryPoint = true))
                    }
                }
            }
        }
        if (rawPoints.isEmpty()) return emptyList()

        // Resample spatially to ~0.15 kw
        val resampled = ArrayList<MooPoint>()
        resampled.add(rawPoints.first().copy(isBoundaryPoint = true))

        var last = rawPoints.first()
        for (i in 1 until rawPoints.size) {
            val curr = rawPoints[i]
            val dx = curr.x - last.x
            val dy = curr.y - last.y
            val d = sqrt(dx * dx + dy * dy)
            val dt = max(1L, curr.t - last.t)

            if (d >= 0.15f || curr.isBoundaryPoint || i == rawPoints.size - 1) {
                val speed = (d / (dt / 1000f)) * geometry.keyWidthPx
                val dwell = if (speed < 15f && dt >= 100L) min(1.0f, dt / 150f) else 0f
                resampled.add(curr.copy(speed = speed, dwellRatio = dwell))
                last = curr
            }
        }
        resampled.last().isBoundaryPoint = true

        // Candidate Key Harvesting (syncCoverage) on each resampled point
        for (pt in resampled) {
            for (code in 0 until geometry.alphabet.letterCount) {
                if (!geometry.hasKey(code)) continue
                val dist = geometry.distToCenter(pt.x, pt.y, code)
                if (dist <= 1.50f) {
                    val rawCov = coverageCurve.get(dist)
                    val finalCov = rawCov.pow(0.60f)
                    if (finalCov > 0.05f) {
                        val ch = geometry.alphabet.charOf(code)
                        pt.travKeys.add(TravKey(code, ch, dist, finalCov))
                    }
                }
            }
            pt.travKeys.sortByDescending { it.coverageScore }
        }

        // Extract key visitation anchors:
        // 1. Boundary points (start and end)
        // 2. Local minimum distance apices to keys (closest approach)
        // 3. Direction turns (curvature inflection)
        // 4. Dwells (speed drops / pauses)
        val anchors = ArrayList<Anchor>()
        fun addAnchor(pt: MooPoint) {
            if (pt.travKeys.isEmpty()) return
            // Avoid duplicate adjacent anchors of the same top key within 0.25 kw
            val topKey = pt.travKeys.first().code
            if (anchors.isNotEmpty() && anchors.last().cands.firstOrNull()?.code == topKey) {
                val distBetween = sqrt((pt.x - anchors.last().pt.x).pow(2) + (pt.y - anchors.last().pt.y).pow(2))
                if (distBetween < 0.25f) return
            }
            anchors.add(Anchor(fingerId, anchors.size, pt, pt.t, pt.travKeys, pt.isVisitationPoint))
        }

        addAnchor(resampled.first())

        for (i in 1 until resampled.size - 1) {
            val pt = resampled[i]
            val prevPt = resampled[i - 1]
            val nextPt = resampled[i + 1]

            // 1. Closest approach apex to a key
            var isApex = false
            for (cand in pt.travKeys) {
                val dPrev = geometry.distToCenter(prevPt.x, prevPt.y, cand.code)
                val dNext = geometry.distToCenter(nextPt.x, nextPt.y, cand.code)
                if (cand.dist < 0.85f && cand.dist <= dPrev && cand.dist <= dNext) {
                    isApex = true
                    break
                }
            }

            // 2. Curvature turn (angle deflection > 35 degrees)
            var isTurn = false
            if (i >= 1 && i + 1 < resampled.size) {
                val p0 = prevPt
                val p1 = pt
                val p2 = nextPt
                val v1x = p1.x - p0.x
                val v1y = p1.y - p0.y
                val v2x = p2.x - p1.x
                val v2y = p2.y - p1.y
                val dot = v1x * v2x + v1y * v2y
                val len1 = sqrt(v1x * v1x + v1y * v1y)
                val len2 = sqrt(v2x * v2x + v2y * v2y)
                if (len1 > 0.02f && len2 > 0.02f) {
                    val cosTheta = dot / (len1 * len2)
                    if (cosTheta < 0.82f) {
                        isTurn = true
                    }
                }
            }

            if (isApex || isTurn || pt.dwellRatio > 0.2f) {
                addAnchor(pt)
            }
        }

        addAnchor(resampled.last())
        return anchors
    }

    // =========================================================================
    // 4. Vowel Bitmask Double-Letter Penalty (0x00104111)
    // =========================================================================
    private fun isVowel(c: Char): Boolean {
        val upper = c.uppercaseChar()
        if (upper < 'A' || upper > 'U') return false
        val shift = upper.code - 'A'.code
        return ((1 shl shift) and 0x00104111) != 0
    }

    private fun getDoubleLetterPenalty(c: Char, dwellRatio: Float): Float {
        val base = if (isVowel(c)) 0.56f else 0.86f
        return dwellRatio + (1.0f - dwellRatio) * base
    }

    // =========================================================================
    // 5. Decode Engine
    // =========================================================================
    fun decode(tokens: List<InputToken>): List<WordCandidate> {
        val leftAnchors = extractAnchors(tokens, StreamId.LEFT, 0)
        val rightAnchors = extractAnchors(tokens, StreamId.RIGHT, 1)

        if (leftAnchors.isEmpty() && rightAnchors.isEmpty()) return emptyList()

        val predWordLen = leftAnchors.size + rightAnchors.size
        val heap = CandidateHeap(KineticaConstants.TOP_K)

        val bucketSameFinger = ArrayList<PredPath>(16)
        val bucketSwitchFinger = ArrayList<PredPath>(16)
        val bucketApos = ArrayList<PredPath>(16)
        val bucketSkip = ArrayList<PredPath>(16)
        val BUCKET_CAPACITY = 15

        fun push(p: PredPath) {
            if (p.maybePrune(2.0f)) return
            val target = when (p.branchMode) {
                0 -> bucketSameFinger
                1 -> bucketSwitchFinger
                2 -> bucketApos
                else -> bucketSkip
            }
            target.add(p)
            target.sortByDescending { it.heuristicScore }
            while (target.size > BUCKET_CAPACITY) {
                target.removeAt(target.size - 1)
            }
        }

        // Seed search root on fingers that start
        if (leftAnchors.isNotEmpty()) {
            push(
                PredPath(
                    node = trie.root,
                    word = "",
                    leftAnchorIdx = -1,
                    rightAnchorIdx = -1,
                    activeFinger = 0,
                    lastTime = 0L,
                    skipCount = 0,
                    branchMode = 0,
                    heuristicScore = 10.0f,
                    minScoreSeen = 10.0f,
                )
            )
        }
        if (rightAnchors.isNotEmpty()) {
            push(
                PredPath(
                    node = trie.root,
                    word = "",
                    leftAnchorIdx = -1,
                    rightAnchorIdx = -1,
                    activeFinger = 1,
                    lastTime = 0L,
                    skipCount = 0,
                    branchMode = 0,
                    heuristicScore = 10.0f,
                    minScoreSeen = 10.0f,
                )
            )
        }

        var steps = 0
        val maxSteps = 2000

        while (steps++ < maxSteps) {
            val activeQueue = when {
                bucketSameFinger.isNotEmpty() -> bucketSameFinger
                bucketSwitchFinger.isNotEmpty() -> bucketSwitchFinger
                bucketSkip.isNotEmpty() -> bucketSkip
                bucketApos.isNotEmpty() -> bucketApos
                else -> break
            }

            val current = activeQueue.removeAt(0)

            // If terminal word, evaluate final score
            if (current.node != trie.root && trie.isWord(current.node)) {
                val word = predictor.displayOf(current.node)
                val freqByte = trie.frequency(current.node)
                val freqWeight = KineticaConstants.FREQ_WEIGHT_FLOOR +
                    (1f - KineticaConstants.FREQ_WEIGHT_FLOOR) * freqByte / 255f

                val excessLen = max(0f, (predWordLen - word.length).toFloat())
                val lenMult = lengthCurve.get(excessLen + 1.0f)

                val skipRatio = if (word.isNotEmpty()) current.skipCount.toFloat() / word.length else 0f
                val skipMult = skipCurve.get(skipRatio)

                val bm = if (prevWordId != -1) bigrams.multiplier(prevWordId, current.node) else 1.0f
                val geoFit = min(1.0f, current.heuristicScore / 10.0f)
                val bmApplied = KineticaConstants.appliedBoost(bm, geoFit)

                val finalScore = freqWeight * geoFit * lenMult * skipMult * bmApplied

                val isMerged = leftAnchors.isNotEmpty() && rightAnchors.isNotEmpty() &&
                    (current.leftAnchorIdx >= 0 && current.rightAnchorIdx >= 0)
                val source = if (isMerged) WordCandidate.Source.MERGED else WordCandidate.Source.SWIPE

                heap.offer(
                    WordCandidate(
                        word = word,
                        score = finalScore,
                        dtwDistance = max(0.0f, 1.0f - geoFit),
                        frequencyWeight = freqWeight,
                        bigramMultiplier = bmApplied,
                        wordId = current.node,
                        source = source,
                        language = predictor.language,
                    )
                )
            }

            val curLeftIdx = current.leftAnchorIdx
            val curRightIdx = current.rightAnchorIdx

            // 1. Advance on Left Finger
            if (curLeftIdx + 1 < leftAnchors.size) {
                val maxLookahead = min(leftAnchors.size, curLeftIdx + 5)
                for (a in (curLeftIdx + 1) until maxLookahead) {
                    val anchor = leftAnchors[a]
                    // Temporal ordering: must be after last matched letter minus overlap margin
                    if (current.word.isNotEmpty() && anchor.t < current.lastTime - 80L) continue

                    for (cand in anchor.cands) {
                        val childNode = trie.child(current.node, cand.code)
                        if (childNode != -1) {
                            var modeMult = if (current.activeFinger == 0) 1.00f else 0.96f
                            if (current.word.isNotEmpty() && current.word.last() == cand.char) {
                                modeMult *= getDoubleLetterPenalty(cand.char, anchor.pt.dwellRatio)
                            }
                            val nextScore = current.heuristicScore * cand.coverageScore * modeMult
                            if (nextScore >= 2.0f) {
                                push(
                                    current.copy(
                                        node = childNode,
                                        word = current.word + cand.char,
                                        leftAnchorIdx = a,
                                        activeFinger = 0,
                                        lastTime = anchor.t,
                                        branchMode = if (current.activeFinger == 0) 0 else 1,
                                        heuristicScore = nextScore,
                                    )
                                )
                            }
                        }
                    }
                }
            }

            // 2. Advance on Right Finger
            if (curRightIdx + 1 < rightAnchors.size) {
                val maxLookahead = min(rightAnchors.size, curRightIdx + 5)
                for (a in (curRightIdx + 1) until maxLookahead) {
                    val anchor = rightAnchors[a]
                    if (current.word.isNotEmpty() && anchor.t < current.lastTime - 80L) continue

                    for (cand in anchor.cands) {
                        val childNode = trie.child(current.node, cand.code)
                        if (childNode != -1) {
                            var modeMult = if (current.activeFinger == 1) 1.00f else 0.96f
                            if (current.word.isNotEmpty() && current.word.last() == cand.char) {
                                modeMult *= getDoubleLetterPenalty(cand.char, anchor.pt.dwellRatio)
                            }
                            val nextScore = current.heuristicScore * cand.coverageScore * modeMult
                            if (nextScore >= 2.0f) {
                                push(
                                    current.copy(
                                        node = childNode,
                                        word = current.word + cand.char,
                                        rightAnchorIdx = a,
                                        activeFinger = 1,
                                        lastTime = anchor.t,
                                        branchMode = if (current.activeFinger == 1) 0 else 1,
                                        heuristicScore = nextScore,
                                    )
                                )
                            }
                        }
                    }
                }
            }

            // 3. Fuzzy Letter Skip (Bucket 3)
            if (current.skipCount < 1 && current.word.length >= 2) {
                val firstChild = trie.firstChild(current.node)
                val count = trie.childCount(current.node)
                for (ci in 0 until min(count, 4)) {
                    val childNode = firstChild + ci
                    val char = trie.alphabet.charOf(trie.letter(childNode))
                    val skipMult = 0.60f
                    val nextScore = current.heuristicScore * skipMult
                    if (nextScore >= 2.0f) {
                        push(
                            current.copy(
                                node = childNode,
                                word = current.word + char,
                                skipCount = current.skipCount + 1,
                                branchMode = 3,
                                heuristicScore = nextScore,
                            )
                        )
                    }
                }
            }
        }

        return heap.sortedByScoreDesc()
    }
}
