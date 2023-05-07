package com.example.tfliteintigration

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.tfliteintigration.ml.OkMissPickModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException


class MainActivity : AppCompatActivity() {
    private var confidenceFactor=0.99
    private lateinit var btnGallery: Button
    private lateinit var btnCamera: Button
    private lateinit var imageView: ImageView
    private lateinit var txtOutput: TextView
    private final val REQUEST_GET_SINGLE_FILE = 101
    private final val REQUEST_IMAGE_CAPTURE = 102
    private var imageBitmap: Bitmap? = null
    val cameraPermission = arrayOf(
        Manifest.permission.CAMERA
    )

    val storagePermission = arrayOf(
        Manifest.permission.READ_EXTERNAL_STORAGE
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnGallery = findViewById(R.id.btn_gallery)
        btnCamera = findViewById(R.id.btn_camera)
        imageView = findViewById(R.id.img)
        txtOutput = findViewById(R.id.txt)

        btnGallery.setOnClickListener {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU){
                val intent = Intent(MediaStore.ACTION_PICK_IMAGES)
                intent.type = "image/*" // or "image/*"
                startActivityForResult(intent, REQUEST_GET_SINGLE_FILE)
            }
            else{
                if (!checkStoragePermission()) {
                    requestStoragePermission()
                } else {
                    val intent = Intent()
                    intent.type = "image/*"
                    intent.action = Intent.ACTION_GET_CONTENT
                    startActivityForResult(
                        Intent.createChooser(intent, "Select Picture"),
                        REQUEST_GET_SINGLE_FILE
                    )

                }
            }

        }
        btnCamera.setOnClickListener {
            if (!checkCameraPermission()) {
                requestCameraPermission()
            } else {
                val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                try {
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                } catch (e: ActivityNotFoundException) {
                    // display error state to the user
                    e.printStackTrace()
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_GET_SINGLE_FILE) {
                val resultUri: Uri? = data!!.data
                try {
                    imageBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, resultUri)
                    imageBitmap = Bitmap.createScaledBitmap(imageBitmap!!, 224, 224, true)
                    imageView.setImageBitmap(imageBitmap)
                    process(imageBitmap)
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
            else if (requestCode == REQUEST_IMAGE_CAPTURE){
                val extras = data!!.extras
                imageBitmap = extras!!["data"] as Bitmap?
                imageBitmap = Bitmap.createScaledBitmap(imageBitmap!!, 224, 224, true)
                imageView.setImageBitmap(imageBitmap)
                process(imageBitmap)
            }
        }
    }


    private fun process(bitmap: Bitmap?) {
        if (bitmap==null){
            return
        }
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val byteBuffer = tensorImage.buffer
        val model = OkMissPickModel.newInstance(applicationContext)

// Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

// Releases model resources if no longer used.
        model.close()
        txtOutput.text ="output: ${if (outputFeature0.floatArray[0]>=confidenceFactor) "OK" else "Miss_Pick"} \n" +
                "Confidence: ${outputFeature0.floatArray[0]}"


// Releases model resources if no longer used.
        model.close()
    }

    private fun requestStoragePermission() {
        ActivityCompat.requestPermissions(this, storagePermission, 2)
    }

    private fun checkStoragePermission(): Boolean {
        val res2 = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.READ_EXTERNAL_STORAGE
        ) == PackageManager.PERMISSION_GRANTED
        return res2
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(this, cameraPermission, 1)
    }

    private fun checkCameraPermission(): Boolean {
        val res3 = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        return res3
    }

}