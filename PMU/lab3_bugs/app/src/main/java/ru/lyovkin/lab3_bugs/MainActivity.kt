package ru.lyovkin.lab3_bugs

import android.os.Bundle
import android.os.Handler
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import java.util.*

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val updateInterval : Long = 1000 / 60

        val gameView = Game(this)
        setContentView(gameView)
        this.window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )

        val handler = Handler()
        val gameTimer = Timer()
        gameTimer.schedule(object : TimerTask() {
            override fun run() {
                handler.post { gameView.invalidate() }
            }
        }, 0, updateInterval)
    }
}