// CoMutableListght (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


package org.pytorch.demo.mnist;

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Color
import android.graphics.PorterDuff
import android.util.AttributeSet
import android.util.Pair
import android.view.MotionEvent
import android.view.View

class HandWrittenDigitView : View {
    private var mPath: Path? = null
    private var mPaint: Paint? = null
    private var mCanvasPaint: Paint? = null;

    private var mCanvas: Canvas? = null
    private var mBitmap: Bitmap? = null
    private var mAllPoints: MutableList<Pair<Float, Float>> = mutableListOf<Pair<Float, Float>>()
    private var mConsecutivePoints: MutableList<Pair<Float, Float>> = mutableListOf<Pair<Float, Float>>()

    constructor(context: Context, attrs: AttributeSet, defStyle: Int) : super(context, attrs, defStyle) {
        mPath = Path()
        mPaint = Paint()
        mPaint!!.color = 0xFF000000.toInt()
        mPaint!!.isAntiAlias = true
        mPaint!!.strokeWidth = 18f
        mPaint!!.style = Paint.Style.STROKE
        mPaint!!.strokeJoin = Paint.Join.ROUND
        mCanvasPaint = Paint(Paint.DITHER_FLAG)
    }

    constructor(context: Context, attrs: AttributeSet) : this(context, attrs, 0) {}

    override fun onDraw(canvas: Canvas) {
        mBitmap?.let {
            canvas.drawBitmap(it, 0f, 0f, mCanvasPaint)
            canvas.drawPath(mPath!!, mPaint!!)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        var x = event.x
        var y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                mConsecutivePoints.clear()
                mConsecutivePoints.add(Pair(x, y))
                mPath!!.moveTo(x, y)
            }
            MotionEvent.ACTION_MOVE -> {
                mConsecutivePoints.add(Pair(x, y))
                mPath!!.lineTo(x, y);
            }
            MotionEvent.ACTION_UP -> {
                mConsecutivePoints.add(Pair(x, y))
                mAllPoints.addAll(mConsecutivePoints)
                mCanvas?.drawPath(mPath!!, mPaint!!)
                mPath?.reset()
            }
            else -> return false;
        }
        invalidate()
        return true
    }

    fun getAllPoints(): MutableList<Pair<Float, Float>> {
        return mAllPoints;
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec)
        if (mBitmap == null) {
            initDrawResources()
        }
    }

    private fun initDrawResources() {
        mBitmap = Bitmap.createBitmap(measuredWidth, measuredHeight, Bitmap.Config.ARGB_8888)
        mCanvas = Canvas(mBitmap!!)
        mCanvasPaint = Paint(Paint.DITHER_FLAG);
    }

    fun clearAllPointsAndRedraw() {
        mCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        mAllPoints.clear()
        mPath!!.reset()

        invalidate()
    }

}
