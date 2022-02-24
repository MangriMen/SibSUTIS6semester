package ru.lyovkin.lab6_widget

import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.w3c.dom.Document
import org.xml.sax.InputSource
import java.io.InputStream
import java.io.StringReader
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.LinkedBlockingQueue
import javax.xml.parsers.DocumentBuilderFactory

class MainActivity : AppCompatActivity() {
    private val cbrURL: String = "https://www.cbr.ru/scripts/xml_metall.asp"
    private val urlCharset = StandardCharsets.UTF_8.name()

    private val metals = mapOf (
        "1" to "Золото",
        "2" to "Серебро",
        "3" to "Платина",
        "4" to "Палладий"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        onUpdate(null)
    }

    fun onUpdate(view: View?) {
        val textView: TextView = findViewById(R.id.textView)
        textView.text = ""
        for ((key, value) in getData()) {
            textView.append("$key: $value\n")
        }
    }

    private fun getData(): MutableMap<String, String> {
        var data = mutableMapOf<String, String>()
        val records = readXml(fetchData()).getElementsByTagName("Record")
        data["Дата"] = records.item(records.length - 1).attributes.getNamedItem("Date").textContent
        for (record in records.length-4 until records.length) {
            val rec = records.item(record)
            data["${metals[rec.attributes.getNamedItem("Code").textContent]}"] = "${rec.childNodes.item(0).textContent}/${rec.childNodes.item(1).textContent}"
        }
        return data
    }

    private fun getDateRequest(): String {
        val dateFormatter = SimpleDateFormat("dd/MM/yyyy", Locale("ru"))
        val calendar = GregorianCalendar()

        val dateEnd = dateFormatter.format(calendar.time)

        calendar.add(Calendar.DAY_OF_MONTH, -10)
        val dateStart = dateFormatter.format(calendar.time)

        return java.lang.String.format(
            "date_req1=%s&date_req2=%s",
            URLEncoder.encode(dateStart, urlCharset),
            URLEncoder.encode(dateEnd, urlCharset)
        )
    }

    private fun fetchData(): String {
        val msgQueue = LinkedBlockingQueue<String>()

        Thread {
            try {
                val connection: HttpURLConnection = URL("$cbrURL?${getDateRequest()}").openConnection() as HttpURLConnection
                connection.setRequestProperty("Accept-Charset", urlCharset)
                val response: InputStream = connection.inputStream

                if (connection.responseCode in 200..399) {
                    Scanner(response).use { scanner ->
                        msgQueue.add(scanner.useDelimiter("\\A").next())
                    }
                }
                else {
                    msgQueue.add("")
                }
            } catch (e: Exception) {
                msgQueue.add("")
                e.printStackTrace()
            }
        }.start()

        return msgQueue.take()
    }

    private fun readXml(xmlString: String): Document {
        val dbFactory = DocumentBuilderFactory.newInstance()
        val dBuilder = dbFactory.newDocumentBuilder()
        val xmlInput = InputSource(StringReader(xmlString))

        return dBuilder.parse(xmlInput)
    }
}