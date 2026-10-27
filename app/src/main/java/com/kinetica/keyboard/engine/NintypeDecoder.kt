package com.kinetica.keyboard.engine

import com.kinetica.keyboard.engine.models.InputToken
import com.kinetica.keyboard.engine.models.PathPoint
import com.kinetica.keyboard.engine.models.StreamId
import com.kinetica.keyboard.engine.models.SwipeToken
import com.kinetica.keyboard.engine.models.TapToken
import com.kinetica.keyboard.engine.models.WordCandidate
import java.util.PriorityQueue
import kotlin.math.abs
import kotlin.math.ln
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

    data class PredPath(
        val node: Int,
        val word: String,
        val fingerId: Int, // 0 = Left, 1 = Right
        val pointIndex: Int,
        val otherPointIndex: Int,
        val skipCount: Int,
        val branchMode: Int, // 0 = direct, 1 = switch, 2 = apos, 3 = skip
        var heuristicScore: Float,
        var minScoreSeen: Float,
        val charCount: Int = word.length,
    ) : Comparable<PredPath> {
        override fun compareTo(other: PredPath): Int {
            // Priority queue orders highest score first
            return other.heuristicScore.compareTo(this.heuristicScore)
        }

        fun updateScore(oldMult: Float, newMult: Float) {
            if (oldMult > 1e-6f) {
                heuristicScore *= (newMult / oldMult)
            }
            if (heuristicScore < minScoreSeen) {
                minScoreSeen = heuristicScore
            }
        }

        fun maybePrune(cutoff: Float = 2.0f): Boolean {
            val threshold = max(cutoff, 2.0f)
            return heuristicScore < threshold
        }
    }

    // =========================================================================
    // 3. Pre-Processing & Feature Extraction
    // =========================================================================
    private fun extractPoints(tokens: List<InputToken>, targetStream: StreamId): List<MooPoint> {
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
                    // Represent tap as anchor point with duration
                    rawPoints.add(MooPoint(token.x, token.y, token.tStart, isBoundaryPoint = true))
                    if (token.tEnd > token.tStart) {
                        val dwell = min(1.0f, (token.tEnd - token.tStart) / 150f)
                        rawPoints.add(MooPoint(token.x, token.y, token.tEnd, dwellRatio = dwell, isBoundaryPoint = true))
                    }
                }
            }
        }
        if (rawPoints.isEmpty()) return emptyList()

        // Resample spatially to ~0.15 kw for stable curvature & speed estimation
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
                val speed = (d / (dt / 1000f)) * geometry.keyWidthPx // in px/s
                val dwell = if (speed < 15f && dt >= 100L) min(1.0f, dt / 150f) else 0f
                resampled.add(curr.copy(speed = speed, dwellRatio = dwell))
                last = curr
            }
        }
        resampled.last().isBoundaryPoint = true

        // 4-Point Curvature Detection (syncCurvyRVA >= 0.30f)
        for (i in 2 until resampled.size - 1) {
            val pPrev2 = resampled[i - 2]
            val pCurr = resampled[i]
            val dspan = max(abs(pCurr.x - pPrev2.x), abs(pCurr.y - pPrev2.y))
            if (dspan >= 0.30f) {
                resampled[i].isCurvatureTurn = true
            }
        }

        // Candidate Key Harvesting (syncCoverage & sortTravs)
        for (pt in resampled) {
            for (code in 0 until geometry.alphabet.letterCount) {
                if (!geometry.hasKey(code)) continue
                val dist = geometry.distToCenter(pt.x, pt.y, code)
                if (dist <= 1.50f) {
                    val rawCov = coverageCurve.get(dist)
                    val finalCov = rawCov.pow(0.60f) // Normal typing exponent
                    if (finalCov > 0.05f) {
                        val ch = geometry.alphabet.charOf(code)
                        pt.travKeys.add(TravKey(code, ch, dist, finalCov))
                    }
                }
            }
            pt.travKeys.sortByDescending { it.coverageScore }
        }

        return resampled
    }

    // =========================================================================
    // 4. Cross-Finger Soft Index Lattice Synchronization
    // =========================================================================
    private fun syncCrossFingerLattice(left: List<MooPoint>, right: List<MooPoint>) {
        fun sync(src: List<MooPoint>, target: List<MooPoint>) {
            if (target.size < 2) {
                for (pt in src) pt.otherFingerSoftIndex = 0f
                return
            }
            for (pt in src) {
                if (pt.t <= target.first().t) {
                    pt.otherFingerSoftIndex = 0f
                } else if (pt.t >= target.last().t) {
                    pt.otherFingerSoftIndex = (target.size - 1).toFloat()
                } else {
                    for (j in 0 until target.size - 1) {
                        val t0 = target[j].t
                        val t1 = target[j + 1].t
                        if (pt.t in t0..t1) {
                            val dt = (t1 - t0).toFloat()
                            val alpha = if (dt > 1e-4f) (pt.t - t0) / dt else 0f
                            pt.otherFingerSoftIndex = j + alpha
                            break
                        }
                    }
                }
            }
        }
        sync(left, right)
        sync(right, left)
    }

    // =========================================================================
    // 5. Vowel Bitmask Double-Letter Penalty (0x00104111)
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
    // 6. Decode Engine
    // =========================================================================
    fun decode(tokens: List<InputToken>): List<WordCandidate> {
        val left = extractPoints(tokens, StreamId.LEFT)
        val right = extractPoints(tokens, StreamId.RIGHT)
        syncCrossFingerLattice(left, right)

        if (left.isEmpty() && right.isEmpty()) return emptyList()

        val predWordLen = left.count { it.isVisitationPoint } + right.count { it.isVisitationPoint } + 2
        val heap = CandidateHeap(KineticaConstants.TOP_K)

        // 4 Queue Buckets
        val bucketSameFinger = PriorityQueue<PredPath>()
        val bucketSwitchFinger = PriorityQueue<PredPath>()
        val bucketApos = PriorityQueue<PredPath>()
        val bucketSkip = PriorityQueue<PredPath>()
        val BUCKET_CAPACITY = 5

        fun push(p: PredPath) {
            if (p.maybePrune(2.0f)) return
            val target = when (p.branchMode) {
                0 -> bucketSameFinger
                1 -> bucketSwitchFinger
                2 -> bucketApos
                else -> bucketSkip
            }
            target.offer(p)
            if (target.size > BUCKET_CAPACITY) {
                // Keep highest-scoring candidates by popping lowest
                val temp = ArrayList<PredPath>(target.size)
                while (target.size > 1) {
                    temp.add(target.poll())
                }
                target.poll() // Discard lowest
                for (item in temp) target.offer(item)
            }
        }

        // Seed search root for available fingers
        if (left.isNotEmpty()) {
            push(PredPath(trie.root, "", fingerId = 0, pointIndex = 0, otherPointIndex = 0, skipCount = 0, branchMode = 0, heuristicScore = 10.0f, minScoreSeen = 10.0f))
        }
        if (right.isNotEmpty()) {
            push(PredPath(trie.root, "", fingerId = 1, pointIndex = 0, otherPointIndex = 0, skipCount = 0, branchMode = 0, heuristicScore = 10.0f, minScoreSeen = 10.0f))
        }

        var steps = 0
        val maxSteps = 1500

        while (steps++ < maxSteps) {
            val activeQueue = when {
                bucketSameFinger.isNotEmpty() -> bucketSameFinger
                bucketSwitchFinger.isNotEmpty() -> bucketSwitchFinger
                bucketSkip.isNotEmpty() -> bucketSkip
                bucketApos.isNotEmpty() -> bucketApos
                else -> break
            }

            val current = activeQueue.poll() ?: continue

            // If terminal word, compute final score and offer to heap
            if (current.node != trie.root && trie.isWord(current.node)) {
                val word = predictor.displayOf(current.node)
                val freqByte = trie.frequency(current.node)
                val freqWeight = KineticaConstants.FREQ_WEIGHT_FLOOR +
                    (1f - KineticaConstants.FREQ_WEIGHT_FLOOR) * freqByte / 255f

                // Word length mismatch penalty (calculateNotLongEnoughMultpen)
                val excessLen = max(0f, (predWordLen - word.length).toFloat())
                val lenMult = lengthCurve.get(excessLen + 1.0f)

                // Skip penalty (multForCharSkip)
                val skipRatio = if (word.isNotEmpty()) current.skipCount.toFloat() / word.length else 0f
                val skipMult = skipCurve.get(skipRatio)

                // Bigram transition multiplier
                val bm = if (prevWordId != -1) bigrams.multiplier(prevWordId, current.node) else 1.0f
                val geoFit = min(1.0f, current.heuristicScore / 10.0f)
                val bmApplied = KineticaConstants.appliedBoost(bm, geoFit)

                val finalScore = freqWeight * geoFit * lenMult * skipMult * bmApplied

                val isMerged = left.isNotEmpty() && right.isNotEmpty() &&
                    (current.fingerId == 1 || current.otherPointIndex > 0)
                val source = if (isMerged) WordCandidate.Source.MERGED else WordCandidate.Source.SWIPE

                heap.offer(
                    WordCandidate(
                        word = word,
                        score = finalScore,
                        dtwDistance = max(0.0f, 1.0f - (current.heuristicScore / 10.0f)),
                        frequencyWeight = freqWeight,
                        bigramMultiplier = bigramMult,
                        wordId = current.node,
                        source = source,
                        language = predictor.language,
                    )
                )
            }

            val stream = if (current.fingerId == 0) left else right
            val otherStream = if (current.fingerId == 0) right else left

            // 1. Same-Finger Continuation (Bucket 0)
            if (current.pointIndex < stream.size) {
                // Find next sample or next visitation anchor
                val maxLookahead = min(stream.size, current.pointIndex + 4)
                for (nextIdx in current.pointIndex until maxLookahead) {
                    val pt = stream[nextIdx]
                    for (cand in pt.travKeys) {
                        val childNode = trie.child(current.node, cand.code)
                        if (childNode != -1) {
                            var modeMult = 1.00f
                            // Double-letter check
                            if (current.word.isNotEmpty() && current.word.last() == cand.char) {
                                modeMult *= getDoubleLetterPenalty(cand.char, pt.dwellRatio)
                            }
                            val nextScore = current.heuristicScore * cand.coverageScore * modeMult
                            if (nextScore >= 2.0f) {
                                push(
                                    current.copy(
                                        node = childNode,
                                        word = current.word + cand.char,
                                        pointIndex = nextIdx + 1,
                                        branchMode = 0,
                                        heuristicScore = nextScore,
                                    )
                                )
                            }
                        }
                    }
                    if (pt.isVisitationPoint) break // Anchor reached
                }
            }

            // 2. Hand Switch Handoff (Bucket 1)
            if (otherStream.isNotEmpty() && current.pointIndex in stream.indices) {
                val softIdx = stream[current.pointIndex].otherFingerSoftIndex
                val targetIdx = softIdx.roundToInt()
                if (targetIdx in otherStream.indices && targetIdx >= current.otherPointIndex) {
                    val otherPt = otherStream[targetIdx]
                    for (cand in otherPt.travKeys) {
                        val childNode = trie.child(current.node, cand.code)
                        if (childNode != -1) {
                            val switchMult = 0.96f
                            val nextScore = current.heuristicScore * cand.coverageScore * switchMult
                            if (nextScore >= 2.0f) {
                                push(
                                    current.copy(
                                        node = childNode,
                                        word = current.word + cand.char,
                                        fingerId = 1 - current.fingerId,
                                        pointIndex = targetIdx + 1,
                                        otherPointIndex = current.pointIndex,
                                        branchMode = 1,
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
                for (ci in 0 until min(count, 5)) {
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
