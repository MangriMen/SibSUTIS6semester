package ru.lyovkin.lab3_bugs

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.text.method.Touch.onTouchEvent
import android.view.MotionEvent
import android.view.View

class Game(context: Context) : View(context) {
    enum class State {
        Play,
        Pause
    }

    private val textStyle : Paint = Paint()
    private var currentState : State = State.Play
    private var msgPause = ""

    init {
        textStyle.color = Color.WHITE
        textStyle.textAlign = Paint.Align.CENTER
        textStyle.textSize = 75F
        textStyle.typeface = Typeface.DEFAULT_BOLD
        textStyle.isAntiAlias = true
    }

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)

        when(currentState) {
            State.Play -> {
                Level.update()
                Level.draw(canvas)
            }
            State.Pause -> {
                canvas?.drawText(
                    msgPause,
                    width / 2F,
                    60F,
                    textStyle
                )
            }
        }
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        if (event?.action == MotionEvent.ACTION_DOWN) {
            return Level.onTouchEvent(event.x, event.y)
        }

        return super.onTouchEvent(event)
    }
}