package ru.lyovkin.calculator

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }

    fun onDigitClick(view: View) {
        var tvMain = findViewById<TextView>(R.id.tvMain)
        var btnDigit = view as Button
        tvMain.append(btnDigit.text)
    }
}