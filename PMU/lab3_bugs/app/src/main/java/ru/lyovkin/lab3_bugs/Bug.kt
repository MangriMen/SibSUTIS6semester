package ru.lyovkin.lab3_bugs

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.media.AudioAttributes
import android.media.SoundPool
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.floor

class Bug(texture_: Bitmap, private var x: Float = 0F, private var y: Float = 0F, context: Context) {
    var isRunning: Boolean = false
    var isAlive: Boolean = true
    private var texture: Bitmap = Bitmap.createScaledBitmap(
        texture_,
        300, 300,
        false
    )
    private var pointsByHit = 10

    private var matrix: Matrix = Matrix()
    private var destX: Float = 0.0f
    private var destY: Float = 0.0f
    private var stepX: Float = 0.0f
    private var stepY: Float = 0.0f
    private var p: Int = 0

    private var hitSoundId: Int
    private val soundPool: SoundPool = SoundPool.Builder().setAudioAttributes(
        AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_GAME)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()
    ).build()

    init { // ??
        matrix.setRotate(0F, texture.width / 2F, texture.height / 2F)
        matrix.reset()
        matrix.postTranslate(x, y)

        hitSoundId = soundPool.load(context, R.raw.hit, 1)
    }

    // ??
    fun isTouched(x_: Float, y_: Float): Boolean {
        return (abs(x - x_ + 60) < 140 && abs(y - y_ + 60) < 150)
    }

    fun hit(): Int {
        isAlive = false
        soundPool.play(hitSoundId, 1F, 1F, 1, 0, 1F)
        return pointsByHit
    }

    fun update(width: Int, height: Int) {
        if (!isRunning) {
            destX = (Math.random() * width).toFloat();
            destY = (Math.random() * height).toFloat();
            stepX = (destX - x) / 57;
            stepY = (destY - y) / 57;

            // ??
            val tp: Int = if (x <= destX && y >= destY)
                floor(Math.toDegrees(atan(abs(x - destX) / abs(y - destY)).toDouble())).toInt()
            else if (x <= destX && y <= destY)
                (90 + floor(Math.toDegrees(atan(abs(y - destY) / abs(x - destX)).toDouble()))).toInt()
            else if (x >= destX && y <= destY)
                (180 + floor(Math.toDegrees(atan(abs(x - destX) / abs(y - destY)).toDouble()))).toInt()
            else
                (270 + floor(Math.toDegrees(atan(abs(y - destY) / abs(x - destX)).toDouble()))).toInt()

            matrix.preRotate((tp - p).toFloat(), texture.width / 2F, texture.height / 2F)
            p = tp;
            isRunning = true;
        } else {
            if (abs(x - destX) < 0.1 &&
                abs(y - destY) < 0.1)
                isRunning = false;

            matrix.postTranslate(stepX, stepY);
            x += stepX;
            y += stepY;
        }
    }

    fun draw(canvas: Canvas?) {
        canvas!!.drawBitmap(texture, matrix, null)
    }
}