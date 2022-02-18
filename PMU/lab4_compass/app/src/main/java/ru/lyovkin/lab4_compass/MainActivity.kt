package ru.lyovkin.lab4_compass

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.animation.Animation
import android.view.animation.LinearInterpolator
import android.view.animation.RotateAnimation
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.math.abs
import kotlin.math.round


class MainActivity : AppCompatActivity() {
    private lateinit var sensorManager: SensorManager
    private lateinit var compassView: ImageView
    private lateinit var angleText: TextView
    private var angle = 0F

    private var floatGravity = FloatArray(3)
    private var floatGeoMagnetic = FloatArray(3)

    private val floatOrientation = FloatArray(3)
    private val floatRotationMatrix = FloatArray(9)

    private lateinit var sensorAccelerometer: Sensor
    private lateinit var sensorMagneticField: Sensor

    private val sensorEventListenerAccelerometer: SensorEventListener = object: SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            floatGravity = event.values

            SensorManager.getRotationMatrix(
                floatRotationMatrix,
                null,
                floatGravity,
                floatGeoMagnetic
            )
            SensorManager.getOrientation(floatRotationMatrix, floatOrientation)

            val newAngle: Float = round((-floatOrientation[0] * 180 / Math.PI)).toFloat()

            val rotateAnimation = RotateAnimation(
                angle,
                newAngle,
                Animation.RELATIVE_TO_SELF,
                0.5F,
                Animation.RELATIVE_TO_SELF,
                0.5F
            )
            rotateAnimation.duration = 300
            rotateAnimation.interpolator = LinearInterpolator()
            rotateAnimation.fillAfter = true
            compassView.startAnimation(rotateAnimation)

            angle = newAngle
            angleText.text = angle.toString()
        }

        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
    }
    private val sensorEventListenerMagneticField: SensorEventListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            floatGeoMagnetic = event.values

            SensorManager.getRotationMatrix(
                floatRotationMatrix,
                null,
                floatGravity,
                floatGeoMagnetic
            )
            SensorManager.getOrientation(floatRotationMatrix, floatOrientation)

            val newAngle: Float = round((-floatOrientation[0] * 180 / Math.PI)).toFloat()

            val rotateAnimation = RotateAnimation(
                angle,
                newAngle,
                Animation.RELATIVE_TO_SELF,
                0.5F,
                Animation.RELATIVE_TO_SELF,
                0.5F
            )
            rotateAnimation.interpolator = LinearInterpolator()
            rotateAnimation.duration = 300
            rotateAnimation.fillAfter = true
            compassView.startAnimation(rotateAnimation)

            angle = newAngle
            angleText.text = angle.toString()
        }

        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        compassView = findViewById(R.id.compassView)
        angleText = findViewById(R.id.angleText)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        sensorAccelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        sensorMagneticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    }

    override fun onResume() {
        super.onResume()

        sensorManager.registerListener(
            sensorEventListenerAccelerometer,
            sensorAccelerometer,
            SensorManager.SENSOR_DELAY_FASTEST
        )
        sensorManager.registerListener(
            sensorEventListenerMagneticField,
            sensorMagneticField,
            SensorManager.SENSOR_DELAY_FASTEST
        )
    }

    override fun onPause() {
        super.onPause()

        sensorManager.unregisterListener(sensorEventListenerAccelerometer)
        sensorManager.unregisterListener(sensorEventListenerMagneticField)
    }
}