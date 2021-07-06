// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


package org.pytorch.demo.mnist

import android.content.Context
import android.os.Bundle
import android.util.Pair
import android.widget.Button
import android.widget.TextView

import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

import java.nio.FloatBuffer

class MainActivity : AppCompatActivity(), Runnable {

    private val TAG = "PT-MNIST"

    private var mModule: Module? = null
    private var mResultTextView: TextView? = null
    private var mRecognizeButton: Button? = null
    private var mClearButton: Button? = null
    private var mDrawView: HandWrittenDigitView? = null

    private val STD = 0.1307f
    private val MEAN = 0.3081f
    private val BLANK = -MEAN / STD
    private val NON_BLANK = (1.0f - MEAN) / STD
    private val MNIST_IMAGE_SIZE = 28

    fun assetFilePath(context: Context, asset: String): String? {
        val file = File(context.filesDir, asset)

        try {
            val inpStream: InputStream = context.assets.open(asset)
            val outStream = FileOutputStream(file, false)
            val buffer = ByteArray(4 * 1024)
            var read: Int

            while (true) {
                read = inpStream.read(buffer)
                if (read == -1) {
                    break
                }
                outStream.write(buffer, 0, read)
            }
            outStream.flush()
            return file.absolutePath
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Error copying asset to file", e)
        }
        return null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mResultTextView = findViewById(R.id.resultTextView)
        mDrawView = findViewById(R.id.drawview)
        mRecognizeButton = findViewById(R.id.recognizeButton)
        mClearButton = findViewById(R.id.clearButton)

        mRecognizeButton!!.setOnClickListener {
            var thread = Thread(this@MainActivity)
            thread.start()
        }

        mClearButton!!.setOnClickListener {
            mResultTextView!!.text = ""
            mDrawView!!.clearAllPointsAndRedraw()
        }

        mModule = LiteModuleLoader.load(assetFilePath(
                this@MainActivity,
                //"mnist.ptl"))
                "mnist_quantized.ptl"))
    }

    override fun run() {
        val result = recognize()
        if (result == -1) {
            return
        }

        runOnUiThread { mResultTextView!!.text = mResultTextView!!.text.toString() + " " + result }
    }

    private fun recognize(): Int {
        var allPoints: MutableList<Pair<Float, Float>> = mDrawView!!.getAllPoints()
        if (allPoints.size == 0) {
            return -1
        }

        var inputs = FloatArray(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE) { _ -> BLANK }

        val width = mDrawView!!.width
        val height = mDrawView!!.height
        for (p: Pair<Float, Float> in allPoints) {
            if (p.first.toInt() > width || p.second.toInt() > height ||
                    p.first.toInt() < 0 || p.second.toInt() < 0) {
                continue
            }
            var x = MNIST_IMAGE_SIZE * p.first.toInt() / width
            var y = MNIST_IMAGE_SIZE * p.second.toInt() / height
            inputs[y * MNIST_IMAGE_SIZE + x] = NON_BLANK
        }

        var inTensorBuffer: FloatBuffer = Tensor.allocateFloatBuffer(28 * 28)
        for (f in inputs) {
            inTensorBuffer.put(f)
        }

        val inTensor = Tensor.fromBlob(inTensorBuffer, longArrayOf(1, 1, 28, 28))

        val outTensor = mModule!!.forward(IValue.from(inTensor)).toTensor()
        val outputs = outTensor.dataAsFloatArray

        var maxScore = -Float.MAX_VALUE
        var maxScoreIdx = -1
        var i = 0
        for (o in outputs) {
            if (o > maxScore) {
                maxScore = o
                maxScoreIdx = i
            }
            i++
        }
        return maxScoreIdx
    }
}