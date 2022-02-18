package ru.lyovkin.lab3_bugs

import android.graphics.*
import android.media.AudioAttributes
import android.media.SoundPool
import android.text.TextPaint
import android.view.MotionEvent
import android.view.View
import kotlin.math.floor

class Level(private val view: View, private val bugsCount: Int) {
    private var bugs: ArrayList<Bug> = ArrayList()
    private val background: Bitmap = BitmapFactory.decodeResource(view.context.resources, R.drawable.background)
    private var score: Long = 30
    private var loseScore: Long = 0
    private var winScore: Long = 200
    private var pointsByMiss: Long = 10

    private var missSoundId: Int
    private val soundPool: SoundPool = SoundPool.Builder().setAudioAttributes(
        AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_GAME)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()
    ).build()

    private val textStyle: TextPaint = TextPaint()
    private val backgroundStyle: Paint = Paint()

    init {
        missSoundId = soundPool.load(view.context, R.raw.miss, 1)

        textStyle.color = Color.WHITE
        textStyle.textAlign = Paint.Align.CENTER
        textStyle.textSize = 75F
        textStyle.typeface = Typeface.DEFAULT_BOLD
        textStyle.isAntiAlias = true

        backgroundStyle.color = Color.DKGRAY
    }

    fun onTouchEvent(event: MotionEvent?) : Boolean {
        var atLeastOneHit = false
        for (bug in bugs) {
            if (bug.isTouched(event!!.x, event.y)) {
                score += bug.hit()
                atLeastOneHit = true
            }
        }

        if (!atLeastOneHit) {
            soundPool.play(missSoundId, 1F, 1F, 1, 0, 1F)
            score -= pointsByMiss
        }

        return true
    }

    private fun generateBug(): Bug {
        val ty: Float
        val tx: Float
        var x = 0F
        var y = 0F

        when (floor(Math.random() * 4).toInt()) {
            0 -> {
                ty = Math.random().toFloat() * view.height
                x = 0F
                y = ty
            }
            1 -> {
                ty = Math.random().toFloat() * view.height
                x = view.width.toFloat()
                y = ty
            }
            2 -> {
                tx = Math.random().toFloat() * view.width
                x = tx
                y = 0F
            }
            3 -> {
                tx = Math.random().toFloat() * view.width
                x = tx
                y = view.height.toFloat()
            }
        }
        return Bug(BitmapFactory.decodeResource(view.context.resources, R.drawable.bug), x, y, view.context)
    }

    private fun updateBugArray() {
        bugs = bugs.filter { it.isAlive } as ArrayList<Bug>

        while(bugs.size < bugsCount) {
            bugs.add(generateBug())
        }
    }

    private fun resetScore() {
        score = 30
    }

    fun update() {
        updateBugArray()

        for (bug in bugs) {
            Thread { bug.update(view.width, view.height) }.start()
        }

        if (score <= loseScore) {
            resetScore()
            Game.pause("Вы проиграли :(")
        }
        else if (score >= winScore) {
            resetScore()
            Game.pause("Вы выиграли!")
        }
    }

    private fun getTextBackgroundSize(x: Float, y: Float, text: String, paint: TextPaint): RectF {
        val fontMetrics: Paint.FontMetrics = paint.fontMetrics
        val halfTextLength: Float = paint.measureText(text) / 2 + 25
        return RectF(((x - halfTextLength)), ((y + fontMetrics.top - 5)), ((x + halfTextLength)), ((y + fontMetrics.bottom + 10)))
    }

    fun draw(canvas: Canvas?) {
        canvas!!.drawBitmap(
            background,
            null,
            Rect(0, 0, view.width, view.height),
            null
        )

        for (bug in bugs) {
            bug.draw(canvas)
        }

        val scoreText = "Score: $score"
        val scoreX = view.width / 2F
        val scoreY = view.height / 16F

        val backgroundRect: RectF = getTextBackgroundSize(scoreX, scoreY, scoreText, textStyle)

        canvas.drawRoundRect(backgroundRect, 16F, 16F, backgroundStyle)
        canvas.drawText(scoreText, scoreX, scoreY, textStyle)
    }
}