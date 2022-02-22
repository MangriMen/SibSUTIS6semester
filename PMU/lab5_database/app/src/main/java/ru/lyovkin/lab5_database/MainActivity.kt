package ru.lyovkin.lab5_database

import android.content.Context
import android.database.Cursor
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.ThreadLocalRandom


class MainActivity : AppCompatActivity() {
    private val dbName: String = "app.db"
    private val dbMode: Int = MODE_PRIVATE

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        findViewById<Button>(R.id.btnGenerate).performClick()
    }

    private fun createTextView(context: Context, text: String, layoutParams: ViewGroup.LayoutParams, gravity: Int, textSize: Float): TextView {
        val tv = TextView(context)
        tv.text = text
        tv.layoutParams = layoutParams
        tv.gravity = gravity
        tv.textSize = textSize
        return tv
    }

    private fun writeRandomStudents() {
        val db = baseContext.openOrCreateDatabase(dbName, dbMode, null)
        db.execSQL("CREATE TABLE IF NOT EXISTS students (name TEXT, weight INTEGER, height INTEGER, age INTEGER);")

        for (i in 1..10) {
            val name = "Kek" + ThreadLocalRandom.current().nextInt(0, 3).toString()
            val weight = ThreadLocalRandom.current().nextInt(50, 100)
            val height = ThreadLocalRandom.current().nextInt(160, 200)
            val age = ThreadLocalRandom.current().nextInt(18, 25)

            db.execSQL("INSERT OR IGNORE INTO students VALUES ('$name', '$weight', '$height', '$age');")
        }

        db.close()
    }

    private fun readAndDisplayDatabase() {
        val tblStudents: TableLayout = findViewById(R.id.tblStudents)
        tblStudents.removeAllViews()

        val params = TableRow.LayoutParams(
            TableRow.LayoutParams.WRAP_CONTENT,
            TableRow.LayoutParams.WRAP_CONTENT,
            1F
        )

        val headerRow = TableRow(this)
        headerRow.addView(createTextView(this, "Имя", params, Gravity.CENTER, 30F))
        headerRow.addView(createTextView(this, "Вес", params, Gravity.CENTER, 30F))
        headerRow.addView(createTextView(this, "Рост", params, Gravity.CENTER, 30F))
        headerRow.addView(createTextView(this, "Возраст", params, Gravity.CENTER, 30F))
        tblStudents.addView(headerRow)

        val db = baseContext.openOrCreateDatabase(dbName, dbMode, null)
        db.execSQL("CREATE TABLE IF NOT EXISTS students (name TEXT, weight INTEGER, height INTEGER, age INTEGER);")

        val query: Cursor = db.rawQuery("SELECT * FROM students;", null)
        while (query.moveToNext()) {
            val newRow = TableRow(this)
            newRow.addView(createTextView(this, query.getString(0), params, Gravity.CENTER, 20F))
            newRow.addView(createTextView(this, query.getInt(1).toString(), params, Gravity.CENTER, 20F))
            newRow.addView(createTextView(this, query.getInt(2).toString(), params, Gravity.CENTER, 20F))
            newRow.addView(createTextView(this, query.getInt(3).toString(), params, Gravity.CENTER, 20F))
            tblStudents.addView(newRow)
        }
        query.close()

        db.close()
    }

    private fun clearDatabase() {
        val db = baseContext.openOrCreateDatabase(dbName, dbMode, null)
        db.execSQL("DROP TABLE IF EXISTS students")
        db.close()
    }

    fun onClear(view: View) {
        clearDatabase()
        readAndDisplayDatabase()
    }

    fun onGenerate(view: View) {
        writeRandomStudents()
        readAndDisplayDatabase()
    }
}