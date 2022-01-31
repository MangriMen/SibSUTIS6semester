package ru.lyovkin.lab1_calculator

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
        val tvMain = findViewById<TextView>(R.id.tvMain)
        val btnDigit = view as Button
        tvMain.append(btnDigit.text)
    }

    fun onClearClick(view: View) {
        val tvMain = findViewById<TextView>(R.id.tvMain)
        tvMain.text = ""
    }

    fun onRemoveClick(view: View) {
        val tvMain = findViewById<TextView>(R.id.tvMain)
        val newText = tvMain.text.substring(0, tvMain.text.length - 1)
        tvMain.text = newText.ifEmpty { "" }
    }

    fun onActionClick(view: View) {
        onDigitClick(view)

    }
}