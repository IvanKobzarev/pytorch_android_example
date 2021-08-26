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
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Pair
import android.widget.Button
import android.widget.TextView

import androidx.appcompat.app.AppCompatActivity
import androidx.core.util.component1
import androidx.core.util.component2
import org.pytorch.*
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Device
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {

  private val TAG = "PT-MNIST"

  private var mModuleCPU_fp32: Module? = null
  private var mModuleNNAPI_fp32: Module? = null
  private var mModuleVulkan_fp32: Module? = null

  private var mTextView: TextView? = null
  private var mRecognizeCpuFp32Button: Button? = null
  private var mRecognizeGpuButton: Button? = null
  private var mRecognizeNnapiFp32Button: Button? = null
  private var mClearButton: Button? = null
  private var mDrawView: HandWrittenDigitView? = null

  private val STD = 0.1307f
  private val MEAN = 0.3081f
  private val BLANK = -MEAN / STD
  private val NON_BLANK = (1.0f - MEAN) / STD
  private val MNIST_IMAGE_SIZE = 28

  private var mBgThread: HandlerThread? = null
  private var mBgHandler: Handler? = null

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
      Log.e(TAG, "Error copying asset to file", e)
    }
    return null
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    mTextView = findViewById(R.id.text)
    mDrawView = findViewById(R.id.draw)
    mRecognizeCpuFp32Button = findViewById(R.id.recognize_cpu)
    mRecognizeGpuButton = findViewById(R.id.recognize_gpu)
    mRecognizeNnapiFp32Button = findViewById(R.id.recognize_nnapi)

    mClearButton = findViewById(R.id.clear)

    mClearButton!!.setOnClickListener {
      mDrawView!!.clearAllPointsAndRedraw()
    }

    mRecognizeCpuFp32Button!!.setOnClickListener {
      mBgHandler!!.post {
        val pair = recognize(mModuleCPU_fp32!!)
          val (digit, timeMs) = pair!!
        runOnUiThread {
          mTextView!!.text = "CPU_fp32 result:$digit $timeMs ms\n" + mTextView!!.text
        }
      }
    }

    mRecognizeGpuButton!!.setOnClickListener {
      mBgHandler!!.post {
        val pair = recognize(mModuleVulkan_fp32!!)
          val (digit, timeMs) = pair!!
        runOnUiThread {
          mTextView!!.text = "GPU_fp32 result:$digit $timeMs ms\n" + mTextView!!.text
        }
      }
    }

    mRecognizeNnapiFp32Button!!.setOnClickListener {
      mBgHandler!!.post {
        val pair = recognize(mModuleNNAPI_fp32!!)
          val (digit, timeMs) = pair!!
        runOnUiThread {
          mTextView!!.text = "NNAPI_fp32 result:$digit $timeMs ms\n" + mTextView!!.text
        }
      }
    }

//        mModuleCPU_fp32 = LiteModuleLoader.load(assetFilePath(
//                this@MainActivity,
//                "mnist.ptl"))
//
//        mModuleNNAPI_fp32 = LiteModuleLoader.load(assetFilePath(
//                this@MainActivity,
//                "mnist-nnapi.ptl"))

    mModuleVulkan_fp32 = LiteModuleLoader.load(assetFilePath(
        this@MainActivity,
        "mnist-vulkan.ptl"),
        null /* extraFiles */,
        Device.VULKAN)

//    mModuleVulkan_fp32 = Module.load(
//        assetFilePath(
//            this@MainActivity,
//            "mnist-vulkan.pt"),
//        null,
//        Device.VULKAN
//    )
    startBgThread()
  }

  override fun onDestroy() {
    stopBgThread()
    super.onDestroy()
  }

  private fun startBgThread() {
    mBgThread = HandlerThread("bgThread")
    mBgThread!!.start()
    mBgHandler = Handler(mBgThread!!.looper)
  }

  private fun stopBgThread() {
    mBgThread!!.quitSafely()
    try {
      mBgThread!!.join()
      mBgThread = null
      mBgHandler = null
    } catch (e: InterruptedException) {
      Log.e(TAG, "Error stopping background thread", e);
    }
  }

  private fun recognize(module: Module): Pair<Int, Long>? {
    val startTimeMillis = SystemClock.elapsedRealtime()
    var allPoints: MutableList<Pair<Float, Float>> = mDrawView!!.getAllPoints()
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

    val inTensor = Tensor.fromBlob(inTensorBuffer, longArrayOf(1, 1, 28, 28), MemoryFormat.CHANNELS_LAST)

    val outTensor = module.forward(IValue.from(inTensor)).toTensor()
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
    val timeMs = SystemClock.elapsedRealtime() - startTimeMillis
    return Pair(maxScoreIdx, timeMs)
  }
}
