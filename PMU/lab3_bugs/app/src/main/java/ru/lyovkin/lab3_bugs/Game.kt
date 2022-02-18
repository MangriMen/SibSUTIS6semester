package ru.lyovkin.lab3_bugs

import android.content.Context
import android.graphics.*
import android.text.method.Touch.onTouchEvent
import android.view.MotionEvent
import android.view.View
import java.lang.Thread.sleep

class Game(context: Context) : View(context) {
    enum class State {
        Play,
        Pause
    }

    private val level: Level = Level(this,5)
    private val textStyle: Paint = Paint()
    private val additionalTextStyle: Paint = Paint()

    companion object {
        private var timer = System.currentTimeMillis()
        private var currentState: State = State.Play
        private var msgPause = ""

        fun pause(msg: String) {
            currentState = State.Pause
            timer = System.currentTimeMillis()
            msgPause = msg
        }
    }

    init {
        textStyle.color = Color.WHITE
        textStyle.textAlign = Paint.Align.CENTER
        textStyle.textSize = 75F
        textStyle.typeface = Typeface.DEFAULT_BOLD
        textStyle.isAntiAlias = true

        additionalTextStyle.color = Color.WHITE
        additionalTextStyle.textAlign = Paint.Align.CENTER
        additionalTextStyle.textSize = 50F
        additionalTextStyle.typeface = Typeface.DEFAULT_BOLD
        additionalTextStyle.isAntiAlias = true

        setBackgroundColor(Color.DKGRAY)
    }

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)

        when(currentState) {
            State.Play -> {
                level.update()
                level.draw(canvas)
            }
            State.Pause -> {
                canvas!!.drawText(
                    msgPause,
                    width / 2F,
                    height / 2F,
                    textStyle
                )

                canvas!!.drawText(
                    "для продолжения нажмите на экран",
                    width / 2F,
                    height - 200F,
                    additionalTextStyle
                )
            }
        }
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        when(currentState) {
            State.Play -> {
                if (event?.action == MotionEvent.ACTION_DOWN) {
                    return level.onTouchEvent(event)
                }
            }
            State.Pause -> {
                if (System.currentTimeMillis() > timer + 450) {
                    currentState = State.Play
                }
            }
        }

        return super.onTouchEvent(event)
    }
}